import argparse
import re
from datetime import date, datetime, timedelta
from pathlib import Path


SNAPSHOT_RE = re.compile(r"^shiftslots_(\d{4}-\d{2}-\d{2})\.xlsx$")


def find_last_working_day(day: date) -> date:
    if day.weekday() == 0:  # Monday: match the app's Friday fallback
        day -= timedelta(days=3)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def retained_dates(today: date) -> set[date]:
    return {
        today,
        today - timedelta(days=1),
        find_last_working_day(today),
        today.replace(day=1),
    }


def cleanup(directory: Path, today: date, protected_names: set[str], dry_run: bool) -> list[Path]:
    keep_dates = retained_dates(today)
    deleted = []

    for path in sorted(directory.glob("shiftslots_*.xlsx")):
        match = SNAPSHOT_RE.match(path.name)
        if not match or path.name in protected_names:
            continue
        file_date = date.fromisoformat(match.group(1))
        if file_date in keep_dates:
            continue
        deleted.append(path)
        if not dry_run:
            path.unlink()

    return deleted


def parse_args():
    parser = argparse.ArgumentParser(description="Delete shift-slot snapshots not used by the app.")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--today", required=True, help="Application date in YYYY-MM-DD format")
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help="Filename that must never be deleted; may be supplied more than once",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    today = datetime.strptime(args.today, "%Y-%m-%d").date()
    if not args.directory.is_dir():
        raise SystemExit(f"Directory does not exist: {args.directory}")

    deleted = cleanup(args.directory, today, set(args.protect), args.dry_run)
    action = "Would delete" if args.dry_run else "Deleted"
    for path in deleted:
        print(f"{action} {path}")
    print(f"{action} {len(deleted)} unused snapshot(s).")


if __name__ == "__main__":
    main()
