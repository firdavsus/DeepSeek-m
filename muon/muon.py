import torch
import torch.distributed as dist
from torch import Tensor
from .matmul_transpose_triton import matmul_transpose_assign


def fast_newtonschulz(G: Tensor, steps: int=5) -> Tensor:
    """
    adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    Arguments:
        G: The gradient or momentum matrix to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None

            # --- EARLY SKIP: if no parameter in this group got a grad, skip whole group ---
            if not any((p.grad is not None) for p in params):
                # nothing to do for this group this step
                continue

            def update_prev():  # optimized Muon implementation
                # safety: if params_world is None, nothing to update
                if params_world is None:
                    return
                if self.world_size > 1:  # only wait when using multi-GPU
                    handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    # g_world here is a view of update_buffer; expected tensor
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)

            # iterate flattened param groups in strides of world_size
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad

                    # If a particular param has no gradient, substitute a zero update
                    # (keeps update_buffer semantics and avoids 'continue' leaving params_world unset)
                    if g is None:
                        # use a zero tensor with same dtype/device as update buffer view
                        g = torch.zeros_like(update_buffer_views[self.rank], device=update_buffer_views[self.rank].device)
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4:  # for conv filters
                        g = g.view(len(g), -1)
                    # orthogonalize (Newton-Schulz)
                    g = fast_newtonschulz(g, steps=group["ns_steps"]).flatten()
                else:
                    # when this rank does not own a param at this base index, we will
                    # use the appropriate view from update_buffer to fill the gather
                    g = update_buffer_views[self.rank]

                if base_i > 0:
                    update_prev()  # flush previous async gather

                if self.world_size == 1:
                    update_buffer.copy_(g)
                else:
                    handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)

                # IMPORTANT: set params_world to the slice that corresponds to the world
                params_world = params[base_i : base_i + self.world_size]

            # final flush for the last chunk
            update_prev()