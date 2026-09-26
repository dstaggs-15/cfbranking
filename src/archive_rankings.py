"""Archive completed ranking outputs without changing ranking computation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path("docs/data")
SOURCE = DATA_ROOT / "rankings.json"
HISTORY_ROOT = DATA_ROOT / "historicals"
INDEX = HISTORY_ROOT / "index.json"


def safe_timestamp(value: str | None) -> tuple[str, str]:
    if value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    captured_at = dt.isoformat().replace("+00:00", "Z")
    filename_stamp = dt.strftime("%Y-%m-%dT%H%M%SZ")
    return captured_at, filename_stamp


def snapshot_metadata(path: Path, existing: dict | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    teams = payload.get("top25") or []
    season = int(payload.get("season") or 0)
    captured_at, _ = safe_timestamp(payload.get("last_build_utc"))

    item = {
        "id": path.stem,
        "season": season,
        "captured_at": captured_at,
        "date": captured_at[:10],
        "file": path.relative_to(DATA_ROOT).as_posix(),
        "team_count": len(teams),
        "top_team": teams[0].get("team") if teams else None,
    }

    if existing:
        for key in ("week", "source_label", "source_url"):
            if existing.get(key) is not None:
                item[key] = existing[key]

    return item


def rebuild_index() -> None:
    snapshots = []
    existing_by_id = {}

    if INDEX.exists():
        try:
            current_index = json.loads(INDEX.read_text(encoding="utf-8"))
            existing_by_id = {
                item.get("id"): item
                for item in current_index.get("snapshots", [])
                if item.get("id")
            }
        except (json.JSONDecodeError, OSError, TypeError):
            existing_by_id = {}

    for path in sorted(HISTORY_ROOT.glob("*/*.json")):
        try:
            snapshots.append(snapshot_metadata(path, existing_by_id.get(path.stem)))
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
            print(f"Skipping unreadable historical snapshot {path}: {exc}")

    snapshots.sort(key=lambda item: item["captured_at"])

    by_season: dict[str, int] = {}
    last_week_by_season: dict[str, int] = {}

    for item in snapshots:
        key = str(item["season"])
        by_season[key] = by_season.get(key, 0) + 1
        item["snapshot_number"] = by_season[key]

        if item.get("week") is not None:
            last_week_by_season[key] = int(item["week"])
        elif key in last_week_by_season:
            last_week_by_season[key] += 1
            item["week"] = last_week_by_season[key]

    index_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "snapshots": snapshots,
    }

    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))

    if not isinstance(payload.get("top25"), list):
        raise SystemExit("rankings.json does not contain a top25 array")

    season = int(payload.get("season") or datetime.now(timezone.utc).year)
    captured_at, filename_stamp = safe_timestamp(payload.get("last_build_utc"))

    season_dir = HISTORY_ROOT / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)

    destination = season_dir / f"{filename_stamp}.json"

    # The snapshot is the completed model output verbatim. No ranking values are changed.
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Archived rankings snapshot: {destination} ({captured_at})")

    rebuild_index()


if __name__ == "__main__":
    main()
