from huggingface_hub import snapshot_download
import os

def download_liveportrait_weights():
    target_dir = "workers/video_worker/liveportrait_repo/pretrained_weights"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Downloading LivePortrait weights to {target_dir}...")
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

if __name__ == "__main__":
    download_liveportrait_weights()
