import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download OWDFA40-Benchmark dataset (zip files only, no extraction)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Target directory, e.g. /path/to/OWDFA40",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="hyzheng/OWDFA40-Benchmark",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download from Hugging Face",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    data_dir = dataset_root / "data"

    dataset_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] repo_id      = {args.repo_id}")

    # Download snapshot (HF cache)
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["data/*.zip", "shape_predictor_68_face_landmarks.dat"],
            local_dir=None,
            local_dir_use_symlinks=True,
            force_download=args.force_download,
        )
    )

    # Copy shape predictor
    predictor_src = snapshot_dir / "shape_predictor_68_face_landmarks.dat"
    predictor_dst = dataset_root / "shape_predictor_68_face_landmarks.dat"
    if not predictor_dst.exists():
        shutil.copy2(predictor_src, predictor_dst)
        print(f"[OK] Copied {predictor_dst.name}")
    else:
        print(f"[SKIP] {predictor_dst.name} already exists")

    # Copy zip files
    zip_files = sorted((snapshot_dir / "data").glob("*.zip"))
    print(f"[INFO] Found {len(zip_files)} zip files")

    for z in zip_files:
        dst = data_dir / z.name
        if dst.exists():
            print(f"[SKIP] {z.name}")
        else:
            shutil.copy2(z, dst)
            print(f"[OK]   {z.name}")

    print("[DONE]")
    print(f"  data_dir = {data_dir}")


if __name__ == "__main__":
    main()
