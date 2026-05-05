import argparse

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset")
    parser.add_argument(
        "--repo-owner",
        default="OliverHausdoerfer",
        help="Hugging Face dataset repository owner",
    )
    parser.add_argument(
        "--dataset-name",
        default="stack_lego_simple_pi05_deploy_2",
        help="Dataset name",
    )
    args = parser.parse_args()

    repo_id = f"{args.repo_owner}/{args.dataset_name}"
    dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
    print(f"Dataset downloaded to: {dataset_path}")


if __name__ == "__main__":
    main()
