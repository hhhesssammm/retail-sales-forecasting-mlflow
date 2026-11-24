import argparse

from src.tracking import configure_tracking, promote


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a reviewed Staging model to Production")
    parser.add_argument("--version", required=True, type=int)
    args = parser.parse_args()
    promote(configure_tracking(), args.version)
    print(f"Version {args.version} is now Production (champion).")
