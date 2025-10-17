from huggingface_hub import HfApi, login

login()

api = HfApi()

api.upload_folder(
    folder_path="ready_model/",
    repo_id="firdavsus/DeepSeek-140M",  # твой репозиторий
    repo_type="model",
    token=""  # указываем токен явно
)

# from huggingface_hub import hf_hub_download

# login()
# # Example: download a specific file from your repo
# local_file = hf_hub_download(
#     repo_id="firdavsus/DeepSeek-140M",   # your repo
#     filename="model-simple-0-0.9064558215047184.pt",        # the specific file you want
#     repo_type="model"                     # type of repo
# )

# print(f"File downloaded to: {local_file}")