from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EVENTS_SUFFIX = "_events.csv"
SPATIOTEMPORAL_SUFFIX = "_spatiotemporal.csv"
KNOWN_ENVIRONMENTS = {
    "tutorial": "Tutorial",
    "house": "House",
    "shoppingcenter": "ShoppingCenter",
    "shoppingcentre": "ShoppingCenter",
    "mall": "ShoppingCenter"
}


@dataclass(slots=True)
class ParticipantFiles:
    participant_id: str
    environments: dict[str, dict[str, list[Path]]] = field(default_factory=dict)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_participants(root: Path) -> list[ParticipantFiles]:
    participants: list[ParticipantFiles] = []
    if not root.exists():
        return participants

    for participant_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        participant = ParticipantFiles(participant_id=participant_dir.name)
        for env_dir in sorted(path for path in participant_dir.iterdir() if path.is_dir()):
            env_name = normalize_environment(env_dir.name)
            if env_name is None:
                continue
            environment_files = collect_environment_files(env_dir)
            if environment_files["events"] or environment_files["trajectory"]:
                participant.environments[env_name] = environment_files

        if participant.environments:
            participants.append(participant)

    return participants


def normalize_environment(name: str) -> str | None:
    return KNOWN_ENVIRONMENTS.get(name.replace(" ", "").lower())


def collect_environment_files(env_dir: Path) -> dict[str, list[Path]]:
    return {
        "events": sorted(env_dir.glob(f"*{EVENTS_SUFFIX}")),
        "trajectory": sorted(env_dir.glob(f"*{SPATIOTEMPORAL_SUFFIX}"))
    }


def load_participant_raw(
    participant: ParticipantFiles,
    *,
    scenario_filter: str | None = None,
    include_tutorial: bool = False,
    include_trajectory: bool = True,
    max_rows_per_file: int = 0,
    longest_session_only: bool = True,
) -> dict[str, Any]:
    environments: dict[str, Any] = {}
    for env_name, env_files in participant.environments.items():
        if scenario_filter and env_name != scenario_filter:
            continue
        if env_name == "Tutorial":
            continue
        include_environment_trajectory = include_trajectory and env_name != "House"
        selected_event_paths = select_longest_session_files(env_files["events"]) if longest_session_only else env_files["events"]

        if env_name == "House":
            trajectory_files: list[dict[str, Any]] = []
        elif include_environment_trajectory:
            selected_trajectory_paths = select_longest_session_files(env_files["trajectory"]) if longest_session_only else env_files["trajectory"]
            trajectory_files = [
                load_raw_csv_text(path, max_rows=max_rows_per_file)
                for path in selected_trajectory_paths
            ]
        else:
            trajectory_files = []

        environments[env_name] = {
            "events": [
                load_raw_csv_text(path, max_rows=max_rows_per_file)
                for path in selected_event_paths
            ],
            "trajectory": trajectory_files,
            "session_selection": {
                "longest_session_only": longest_session_only,
                "available_event_files": [path.name for path in env_files["events"]],
                "selected_event_files": [path.name for path in selected_event_paths],
                "available_trajectory_files": [path.name for path in env_files["trajectory"]],
                "selected_trajectory_files": [
                    file_info.get("file_name", "")
                    for file_info in trajectory_files
                ],
                "selection_rule": "When multiple CSVs exist for a scenario, Python keeps only the file with the maximum timestamp duration for each data kind.",
            },
        }

    return {
        "participant_id": participant.participant_id,
        "environments": environments
    }


def load_raw_csv_text(path: Path, *, max_rows: int = 0) -> dict[str, Any]:
    lines: list[str] = []
    profile = csv_time_profile(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        output_columns = ["row_number", *fieldnames]
        lines.append(csv_line(output_columns))
        for row_number, row in enumerate(reader, start=2):
            if max_rows and len(lines) > max_rows:
                break
            lines.append(csv_line([row_number, *[row.get(column, "") for column in fieldnames]]))

    return {
        "file_name": path.name,
        "file_path": str(path),
        "columns": fieldnames,
        "included_data_row_count": max(0, len(lines) - 1),
        "truncated": bool(max_rows and len(lines) > max_rows),
        "source_row_count": profile["row_count"],
        "first_timestamp": profile["first_timestamp"],
        "last_timestamp": profile["last_timestamp"],
        "duration_ms": profile["duration_ms"],
        "raw_csv_with_row_numbers": "\n".join(lines)
    }


def select_longest_session_files(paths: list[Path]) -> list[Path]:
    if len(paths) <= 1:
        return paths
    scored = []
    for path in paths:
        profile = csv_time_profile(path)
        scored.append((
            profile["duration_ms"],
            profile["row_count"],
            path.name,
            path,
        ))
    longest = max(scored, key=lambda item: (item[0], item[1], item[2]))
    return [longest[3]]


def csv_time_profile(path: Path) -> dict[str, Any]:
    timestamps: list[int] = []
    row_count = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            timestamp = row.get("time") or row.get("UnixTime")
            parsed = safe_int(timestamp)
            if parsed is not None:
                timestamps.append(parsed)
    if not timestamps:
        return {
            "row_count": row_count,
            "first_timestamp": None,
            "last_timestamp": None,
            "duration_ms": 0,
        }
    first = min(timestamps)
    last = max(timestamps)
    return {
        "row_count": row_count,
        "first_timestamp": first,
        "last_timestamp": last,
        "duration_ms": last - first,
    }


def safe_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def csv_line(values: list[Any]) -> str:
    return ",".join(csv_cell(value) for value in values)


def csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n", "\r"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def load_raw_batch(
    participants_root: Path,
    *,
    participant_filter: str | None = None,
    scenario_filter: str | None = None,
    max_participants: int = 0,
    include_tutorial: bool = False,
    include_trajectory: bool = True,
    max_rows_per_file: int = 0,
    longest_session_only: bool = True,
) -> dict[str, Any]:
    discovered = discover_participants(participants_root)
    selected = []
    for participant in discovered:
        if participant_filter and participant.participant_id != participant_filter:
            continue
        selected.append(participant)
        if max_participants and len(selected) >= max_participants:
            break

    return {
        "participants_root": str(participants_root),
        "participant_count": len(selected),
        "participants": [
            load_participant_raw(
                participant,
                scenario_filter=scenario_filter,
                include_tutorial=include_tutorial,
                include_trajectory=include_trajectory,
                max_rows_per_file=max_rows_per_file,
                longest_session_only=longest_session_only,
            )
            for participant in selected
        ]
    }
