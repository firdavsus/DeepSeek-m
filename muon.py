# combined_muon_adam.py
import math
import torch
import torch.distributed as dist
from torch import Tensor
import triton
import triton.language as tl

# ---------------------------
# Triton matmul-transpose kernel + helper (copied/adapted from your snippet)
# ---------------------------
def get_autotune_config():
    return [triton.Config({'BLOCK_SIZE_M': blk_m, 'BLOCK_SIZE_K': blk_k, 'GROUP_SIZE_M': grp_sz},
                          num_stages=n_stages, num_warps=n_warps)
            for blk_m in [32, 64, 128]
            for blk_k in [32, 64]
            for grp_sz in [8]
            for n_stages in [3, 4, 5]
            for n_warps in [4, 8]
            ]

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'K'],
)
@triton.jit
def mmt_kernel(
        x, y,
        M, K,
        stride_xm, stride_xk,
        stride_ym, stride_yn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m > pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk

    c = accumulator.to(x.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)

def matmul_transpose_assign(d_in, d_out):
    assert d_in.is_cuda and d_out.is_cuda
    assert d_in.device == d_out.device
    assert d_in.ndim == 2 and d_out.ndim == 2
    M, K = d_in.shape
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(M, META['BLOCK_SIZE_M']), )
    with torch.cuda.device(d_in.device.index):
        mmt_kernel[grid](
            d_in,
            d_out,
            M,
            K,
            d_in.stride(0),
            d_in.stride(1),
            d_out.stride(0),
            d_out.stride(1)
        )

def matmul_transpose(d_in):
    M, _ = d_in.shape
    d_out = torch.empty((M, M), device=d_in.device, dtype=d_in.dtype)
    matmul_transpose_assign(d_in, d_out)
    return d_out

def fast_newtonschulz(G: Tensor, steps: int = 5) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = False
    if G.size(-2) > G.size(-1):
        X = X.mT
        transposed = True

    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X.to(G.dtype)

# ---------------------------
# Combined optimizer
# ---------------------------
class Muon(torch.optim.Optimizer):
    """
    Combined optimizer:
      - groups with 'use_muon': Muon-style fast Newton–Schulz orthogonalized update
      - groups without 'use_muon': Adam update

    param_groups: list of dicts with flag 'use_muon' present for every group.
      If use_muon == True, group may include:
         lr, momentum, weight_decay, nesterov (optional), ns_steps (optional)
      If use_muon == False, group may include:
         lr, betas, eps, weight_decay
    rank, world_size: for distributed usage (default rank=0, world_size=1 -> single GPU)
    """
    def __init__(self, param_groups, rank: int = 0, world_size: int = 1):
        # validate + set defaults similar to your SingleDeviceMuonWithAuxAdam
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                group["nesterov"] = group.get("nesterov", True)
                group["ns_steps"] = group.get("ns_steps", 5)
                # allowed keys: params, lr, momentum, weight_decay, use_muon, nesterov, ns_steps
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
                # allowed keys: params, lr, betas, eps, weight_decay, use_muon
        # convert muon groups to grouped buffers like original Muon: group by param.numel()
        self.rank = rank
        self.world_size = world_size
        # build param_groups for torch.Optimizer; for muon groups we will add update_buffer entries
        processed = []
        for group in param_groups:
            if group["use_muon"]:
                params = [p for p in group["params"]]
                params = list(params)
                sizes = {p.numel() for p in params}
                # create sub-groups per size so we can have contiguous buffers (like Muon)
                for size in sizes:
                    sub_params = [p for p in params if p.numel() == size]
                    # create update_buffer tensor shared across world_size
                    # use bfloat16 to match Muon
                    device = sub_params[0].device
                    buf = torch.empty((self.world_size, size), dtype=torch.bfloat16, device=device)
                    sub_group = {
                        "params": sub_params,
                        "update_buffer": buf,
                        "update_buffer_views": [buf[i] for i in range(self.world_size)],
                        # copy muon-specific defaults
                        "lr": group["lr"],
                        "momentum": group["momentum"],
                        "weight_decay": group["weight_decay"],
                        "nesterov": group.get("nesterov", True),
                        "ns_steps": group.get("ns_steps", 5),
                        "use_muon": True
                    }
                    processed.append(sub_group)
            else:
                processed.append(group)
        super().__init__(processed, defaults={})
        # no extra state fields here beyond param state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                # Muon-style fast path
                update_buffer = group["update_buffer"]
                update_buffer_views = group["update_buffer_views"]
                params = group["params"]
                handle = None
                params_world = None

                def update_prev():
                    # apply the previously-gathered updates to params_world
                    nonlocal handle, params_world, update_buffer_views
                    if handle is not None:
                        handle.wait()
                    for p_world, g_world in zip(params_world, update_buffer_views):
                        # weight decay + param update
                        p_world.mul_(1 - group["lr"] * group["weight_decay"])
                        scale = (max(1, p_world.size(-2)/p_world.size(-1)) ** 0.5)
                        p_world.add_(g_world.view_as(p_world), alpha=-group["lr"] * scale)

                # iterate over params in chunks of world_size (matching Muon logic)
                for base_i in range(0, len(params), self.world_size):
                    p = None
                    if base_i + self.rank < len(params):
                        p = params[base_i + self.rank]
                        g = p.grad
                        if g is None:
                            continue
                            # raise RuntimeError("Muon-group param has no grad")
                        state = self.state[p]
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(g)
                        buf = state["momentum_buffer"]
                        # buf = buf * momentum + g * (1 - momentum)  -> using lerp
                        buf.lerp_(g, 1 - group["momentum"])
                        # compute effective gradient (nesterov if requested)
                        if group.get("nesterov", True):
                            g_eff = g.lerp(buf, group["momentum"])
                        else:
                            g_eff = buf
                        # flatten & reshape for NS: if 4D conv filters -> flatten spatial dims
                        g_work = g_eff
                        if g_work.ndim == 4:
                            g_work = g_work.view(len(g_work), -1)
                        # run Newton-Schulz orthogonalization (returns same dtype as input)
                        g_ns = fast_newtonschulz(g_work, steps=group.get("ns_steps", 5)).flatten()
                    else:
                        # if this rank doesn't own a param in this slot, use a view into the shared buffer
                        g_ns = update_buffer_views[self.rank]

                    if base_i > 0 and self.world_size > 1:
                        update_prev()

                    params_world = params[base_i: base_i + self.world_size]

                    if self.world_size > 1:
                        # all-gather into update_buffer (async)
                        handle = dist.all_gather_into_tensor(update_buffer, g_ns, async_op=True)
                    else:
                        # single-device: apply update immediately
                        if p is not None:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                            scale = (max(1, p.size(-2)/p.size(-1)) ** 0.5)
                            p.add_(g_ns.view_as(p), alpha=-group["lr"] * scale)

                if self.world_size > 1:
                    update_prev()

            else:
                # Adam path
                for p in group["params"]:
                    if p.grad is None:
                        # emulate behavior in your slow impl: create zero grad to avoid sync issues
                        p.grad = torch.zeros_like(p)
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    eps = group["eps"]

                    # update biased first and second moment estimates
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    step = state["step"]
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step

                    # computing Adam update (as a direction, lr applied outside)
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    step_size = math.sqrt(bias_correction2) / bias_correction1
                    update = exp_avg / denom
                    update = update * step_size  # scale for bias correction

                    # weight decay as in original snippet: multiplicative before add
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
