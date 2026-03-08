from huggingface_hub import HfApi

api = HfApi(token="")

api.upload_folder(
    folder_path="data/cpt_packed",
    repo_id="lenguyen1807/cpt_packed",
    repo_type="dataset",
    commit_message="Sync local data folder",
)
