#!/usr/bin/env python3
"""Download JeffSackmann tennis datasets to data/raw/.

Clones (or updates) the following repositories:
    tennis_atp            — ATP match-level data (rankings, stats)
    tennis_wta            — WTA match-level data
    tennis_pointbypoint   — Point-by-point sequences (ATP/WTA)
    tennis_slam_pointbypoint — Grand Slam point-by-point (2011+)

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --update   # git pull existing repos
"""
import argparse
import os
import subprocess
import sys

REPOS = {
    "tennis_atp": "https://github.com/JeffSackmann/tennis_atp.git",
    "tennis_wta": "https://github.com/JeffSackmann/tennis_wta.git",
    "tennis_pointbypoint": "https://github.com/JeffSackmann/tennis_pointbypoint.git",
    "tennis_slam_pointbypoint": "https://github.com/JeffSackmann/tennis_slam_pointbypoint.git",
}

DATA_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def clone_or_update(name: str, url: str, update: bool) -> None:
    dest = os.path.join(DATA_RAW_DIR, name)
    if os.path.exists(dest):
        if update:
            print(f"  Updating {name}...")
            result = subprocess.run(["git", "-C", dest, "pull", "--ff-only"], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  WARNING: git pull failed for {name}: {result.stderr.strip()}")
            else:
                print(f"  {name}: {result.stdout.strip() or 'already up to date'}")
        else:
            print(f"  {name}: already exists (use --update to refresh)")
    else:
        print(f"  Cloning {name} from {url}...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR cloning {name}: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"  {name}: cloned OK")


def main():
    parser = argparse.ArgumentParser(description="Download JeffSackmann tennis datasets")
    parser.add_argument("--update", action="store_true", help="Update existing repos with git pull")
    parser.add_argument("--repos", nargs="+", choices=list(REPOS), default=list(REPOS),
                        help="Which repos to download (default: all)")
    args = parser.parse_args()

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    print(f"Data directory: {DATA_RAW_DIR}")

    for name in args.repos:
        clone_or_update(name, REPOS[name], update=args.update)

    print("\nDone. Next step: python scripts/build_features.py")


if __name__ == "__main__":
    main()
