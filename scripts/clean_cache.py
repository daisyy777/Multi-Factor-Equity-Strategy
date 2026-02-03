"""
Script to clean cache files and reset the project for a fresh start.

This script removes:
- Cached price data (data/raw/*.parquet)
- Cached fundamental data (data/raw/*.parquet)
- Processed data (data/processed/*)
- Previous results (results/*)
- Previous plots (reports/figures/*)
- Log files (logs/*.log)

Usage:
    python scripts/clean_cache.py
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR
from pathlib import Path

def clean_cache(confirm: bool = True):
    """
    Clean all cache files and previous results.
    
    Parameters
    ----------
    confirm : bool
        If True, ask for confirmation before deleting
    """
    print("=" * 60)
    print("CLEANING CACHE AND RESETTING PROJECT")
    print("=" * 60)
    
    # Directories to clean
    dirs_to_clean = {
        "Price & Fundamental Cache": RAW_DATA_DIR,
        "Processed Data": PROCESSED_DATA_DIR,
        "Results": RESULTS_DIR,
        "Figures": FIGURES_DIR,
    }
    
    # Log directory
    log_dir = project_root / "logs"
    
    total_files = 0
    total_size = 0
    
    # Count files first
    for name, dir_path in dirs_to_clean.items():
        if dir_path.exists():
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
    
    # Count log files
    if log_dir.exists():
        for file_path in log_dir.glob("*.log"):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
    
    if total_files == 0:
        print("\n[OK] No cache files found. Project is already clean!")
        return
    
    print(f"\nFound {total_files} files to delete ({total_size / 1024 / 1024:.2f} MB)")
    
    if confirm:
        response = input("\nDo you want to delete all cache files? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled. No files were deleted.")
            return
    
    deleted_count = 0
    deleted_size = 0
    
    # Delete files
    for name, dir_path in dirs_to_clean.items():
        if dir_path.exists():
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        deleted_size += size
                        print(f"  [OK] Deleted: {file_path.name}")
                    except Exception as e:
                        print(f"  [ERROR] Error deleting {file_path.name}: {e}")
    
    # Delete log files
    if log_dir.exists():
        for file_path in log_dir.glob("*.log"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    deleted_size += size
                    print(f"  [OK] Deleted: {file_path.name}")
                except Exception as e:
                    print(f"  [ERROR] Error deleting {file_path.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"[OK] Cleanup complete!")
    print(f"  Deleted {deleted_count} files ({deleted_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)
    print("\nYou can now run a fresh backtest with:")
    print("  python scripts/run_backtest.py")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean cache files and reset project")
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    clean_cache(confirm=not args.yes)
