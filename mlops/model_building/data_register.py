from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "lcsekar/bank-customer-churn"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' does not exist. Creating it.")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_folder(
    folder_path="mlops",
    repo_id=repo_id,
    repo_type=repo_type,
)
