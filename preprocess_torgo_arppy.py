import argparse
from pathlib import Path
import shutil


def read_problematic_files(bad_files_path):
    """Read problematic files from the bad_utts_torgo file"""
    problematic = set()
    with open(bad_files_path, 'r') as f:
        for line in f:
            # Remove comments and strip whitespace
            clean_line = line.split('#')[0].strip()
            if clean_line:  # Only add if there's content
                problematic.add(clean_line)
    return problematic


def copy_good_files(source_dir, dest_dir, bad_files):
    """Copy files not in the problematic list"""
    copied_files = 0
    for file in source_dir.glob("*"):
        if file.is_file():
            stem = file.stem
            if stem not in bad_files:
                shutil.copy2(file, dest_dir / file.name)
                copied_files += 1
                print(f"Copied: {file.name}")
    return copied_files


def preprocess_torgo(sr, source_dir, dest_dir):
    # Define paths
    bad_files_path = Path("bad_utts_torgo")  # Assuming it's in current directory

    # Read problematic files
    problematic_files = read_problematic_files(bad_files_path)
    print(f"Found {len(problematic_files)} problematic files in the list")

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Copy good files
    copied_count = copy_good_files(source_dir, dest_dir, problematic_files)
    print(f"\nDone! Copied {copied_count} good files to {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Torgo dataset.")
    parser.add_argument("--base_dir", required=True, help="Path to the base directory of the Torgo dataset.", type=Path)
    parser.add_argument("--target_dir", required=True, help="Path to the target directory for processed data.", type=Path)
    args = parser.parse_args()

    SAMPLE_RATE = 16000

    preprocess_torgo(SAMPLE_RATE, args.base_dir, args.target_dir)