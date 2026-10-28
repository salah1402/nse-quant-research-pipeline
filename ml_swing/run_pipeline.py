"""
Orchestration script for Phase 1 & 2: Data Download & Feature Engineering.
"""
import sys
import os

from download_data import download_all_data
from build_features import process_features_and_target

def main():
    print("=" * 60)
    print("STEP 1: UNIVERSE SELECTION & UPSTOX DATA DOWNLOAD")
    print("=" * 60)
    download_all_data()

    print("\n" + "=" * 60)
    print("STEP 2: FEATURE ENGINEERING & DATASET PREPARATION")
    print("=" * 60)
    df = process_features_and_target()

    if df is not None:
        print("\nPipeline execution completed successfully.")
    else:
        print("\nPipeline execution encountered errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
