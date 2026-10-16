"""
Upload release artifacts to Hugging Face Model Hub.
Reads token exclusively from the HF_TOKEN environment variable.
Never commits or logs the token.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload Vietnamese license plate benchmark models to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="neitpm/vn-plate-detection-benchmark",
        help="Target repository ID on Hugging Face"
    )
    parser.add_argument(
        "--release_dir",
        type=str,
        default="hf_staging",
        help="Local release directory containing model artifacts and metadata"
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token or not token.strip():
        print("ERROR: HF_TOKEN environment variable is not set or empty.", file=sys.stderr)
        print("Please set your Hugging Face write token before running:", file=sys.stderr)
        print("  $env:HF_TOKEN = 'hf_...'", file=sys.stderr)
        sys.exit(1)

    release_path = Path(args.release_dir).resolve()
    if not release_path.exists():
        print(f"ERROR: Staging directory '{release_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Hugging Face Hub with repository: {args.repo_id}...")
    api = HfApi(token=token.strip())

    # Create model repository if it doesn't already exist
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"Repository '{args.repo_id}' ready on Hugging Face Hub.")
    except Exception as e:
        print(f"Repository creation message / warning: {e}", file=sys.stderr)

    # Upload all files from the staging directory
    print(f"Uploading files from '{release_path}' to '{args.repo_id}'...")
    uploaded_files = []
    for item in release_path.rglob("*"):
        if item.is_file():
            path_in_repo = str(item.relative_to(release_path)).replace("\\", "/")
            print(f"  Uploading: {path_in_repo} ({item.stat().st_size / (1024 * 1024):.2f} MB)...")
            api.upload_file(
                path_or_fileobj=str(item),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Upload {path_in_repo}"
            )
            uploaded_files.append(path_in_repo)

    print(f"\nUpload complete! Total files uploaded: {len(uploaded_files)}")
    for f in uploaded_files:
        print(f"  - {f}")

if __name__ == "__main__":
    main()
