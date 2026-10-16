from __future__ import annotations

import csv
import math
import unicodedata
from collections import defaultdict
from io import StringIO
from typing import Any


SHOP_ALIASES = {
    "Decathlon": ["decathlon", "décathlon"],
    "Fnac": ["fnac"],
    "Zara": ["zara", "zara1", "zara2"],
}

ITEM_ALIASES = {
    "sandals": [
        "sandals",
        "sandales",
        "sandale",
        "sandals(clone)",
        "sandales(clone)",
        "sandale(clone)",
        "slippers",
        "slippers(clone)",
    ],
    "guide de voyage": [
        "guide de voyage",
        "guide voyage",
        "guide",
        "livre",
        "livre a la fnac",
        "travel guide",
        "travelguide",
        "travelguide(clone)",
        "trip guide",
        "tripguide",
        "tripguide(clone)",
        "guidevoyage",
        "guidevoyage(clone)",
        "guide_voyage",
    ],
    "casquette": [
        "casquette",
        "cap",
        "cap(clone)",
        "hat",
        "casquette(clone)",
        "hat(clone)",
    ],
}

STRICT_TARGET_ITEM_ALIASES = {
    "sandals": ["sandale", "sandales"],
    "guide de voyage": ["guide de voyage", "guide voyage"],
    "casquette": ["casquette"],
}

SUPPORTING_ITEM_OBJECT_ALIASES = {
    "sandals": [
        "slippers",
        "slippers(clone)",
        "sandals",
        "sandals(clone)",
        "sandale(clone)",
        "sandales(clone)",
    ],
    "guide de voyage": [
        "tripguide",
        "tripguide(clone)",
        "travelguide",
        "travelguide(clone)",
        "guidevoyage",
        "guidevoyage(clone)",
    ],
    "casquette": [
        "hat",
        "hat(clone)",
        "cap",
        "cap(clone)",
        "casquette(clone)",
    ],
}

RELATED_OR_WRONG_ITEM_ALIASES = {
    "sandals": ["chaussure", "chaussures"],
    "guide de voyage": ["livre", "book"],
    "casquette": [],
}

INTENTION_ALIASES = {
    "sandals": ["sandales", "sandals"],
    "guide de voyage": ["guide_voyage", "travel_guide", "guide voyage", "guide de voyage"],
    "casquette": ["casquette", "cap", "hat"],
}

WATCH_VERBS = {"starttowatch", "stoptowatch", "watched", "watch"}
BUYING_WORDS = ("ask", "pay", "iteminbag", "bag", "validate", "interact")
DELAY_THRESHOLD_SECONDS = 120


def calculate_python_cues(
    *,
    raw_batch: dict[str, Any],
    tasks_config: dict[str, Any],
    clues_config: dict[str, Any],
    reference_batch: dict[str, Any] | None = None,
    shop_locations_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    shop_position_model = build_shop_position_model(
        reference_batch or raw_batch,
        shop_locations_config=shop_locations_config or tasks_config.get("shop_locations")
    )
    participants = []
    for participant in raw_batch.get("participants", []):
        participant_events = collect_events(participant)
        participant_trajectory = collect_trajectory_data(participant)
        task_results = []
        for task in enabled_event_tasks(tasks_config):
            task_results.append(
                analyze_task(
                    participant_events,
                    participant_trajectory,
                    task,
                    clues_config,
                    shop_position_model
                )
            )
        participants.append({
            "participant_id": participant.get("participant_id"),
            "evidence_summary": participant_evidence_summary(task_results),
            "task_results": task_results,
            "trajectory_summary": public_trajectory_summary(participant_trajectory),
        })

    return {
        "analysis_type": "python_calculated_cues_for_llm_reasoning",
        "important_note": (
            "Python calculated clue evidence conservatively from raw CSV rows. "
            "It also inferred shop-area positions from all participants' ShoppingCenter Enter/Exit events matched to trajectory rows. "
            "The LLM should reason over this evidence, not invent missing events."
        ),
        "trajectory_shop_position_model": shop_position_model,
        "participants": participants,
    }


def python_cues_markdown(cue_batch: dict[str, Any]) -> str:
    lines = [
        "# Python Clue Analysis Report",
        "",
        "Python calculated clue evidence from the raw CSV rows. This is not an LLM report.",
        "",
    ]
    model = cue_batch.get("trajectory_shop_position_model", {})
    shops = model.get("shops", {})
    if shops:
        lines.extend(["## Inferred Shop Positions", ""])
        for shop_name, shop_model in shops.items():
            center = shop_model.get("representative_center") or {}
            lines.append(
                f"- {shop_name}: representative `{shop_model.get('representative_variant')}`, "
                f"center x={center.get('x')}, y={center.get('y')}, z={center.get('z')}, "
                f"radius={shop_model.get('representative_radius')}"
            )
        lines.append("")

    for participant in cue_batch.get("participants", []):
        lines.extend([f"## {participant.get('participant_id', 'unknown participant')}", ""])
        evidence_summary = participant.get("evidence_summary", {})
        if evidence_summary:
            lines.extend(["Evidence:", ""])
            lines.append("| Evidence | Count / value |")
            lines.append("|---|---:|")
            for label, value in evidence_summary.items():
                lines.append(f"| {label} | {value} |")
            lines.append("")
        for task_result in participant.get("task_results", []):
            task = task_result.get("task", {})
            lines.extend([
                f"### {task.get('example_task', 'unknown task')}",
                "",
                f"- Python task status: `{task_result.get('python_task_status')}`",
            ])
            completion_evidence = task_result.get("completion_evidence", {})
            if completion_evidence:
                lines.extend(["", "Completion evidence:"])
                for label, events in completion_evidence.items():
                    if events:
                        lines.append(f"- {label}:")
                        for event in events[:3]:
                            citation = event_citation(event)
                            raw_values = citation.get("raw_values") or []
                            if raw_values:
                                lines.append(f"  - {raw_values[0]}")
                            else:
                                lines.append(
                                    f"  - {citation.get('file_name')}, rows {citation.get('row_numbers')}, "
                                    f"time={citation.get('in_game_time') or citation.get('time')}, "
                                    f"verb={citation.get('verb')}, object={citation.get('object')}"
                                )

            trajectory = task_result.get("trajectory_evidence", {})
            if trajectory:
                lines.extend([
                    "- Trajectory evidence: "
                    + f"`{trajectory.get('status')}`; nearest distance={trajectory.get('nearest_distance')}; "
                    + f"estimated time near={trajectory.get('estimated_time_near_seconds')} seconds; "
                    + f"representative shop={trajectory.get('representative_shop_variant')}; "
                    + f"inferred target floor={trajectory.get('inferred_target_floor')}; "
                    + f"floors seen={trajectory.get('floors_seen_in_shoppingcenter')}",
                    f"- Trajectory note: {task_result.get('trajectory_note')}",
                ])

            house_recall = task_result.get("house_recall_evidence", {})
            if house_recall:
                lines.append(
                    "- House recall evidence: "
                    + f"available={house_recall.get('house_logs_available')}, "
                    + f"mentioned={house_recall.get('mentioned')}, "
                    + f"confirmed={house_recall.get('confirmed')}, "
                    + f"confused={house_recall.get('confused')}"
                )
                house_examples = (
                    house_recall.get("confirmed_events")
                    or house_recall.get("confused_events")
                    or house_recall.get("mentioned_events")
                    or []
                )
                if house_examples:
                    lines.append("- House evidence examples:")
                    for event in house_examples[:3]:
                        citation = event_citation(event)
                        raw_values = citation.get("raw_values") or []
                        for raw_value in raw_values[:4]:
                            lines.append(f"  - {raw_value}")

            supported = [cue for cue in task_result.get("cue_results", []) if cue.get("status") == "supported"]
            rejected_or_unclear = [cue for cue in task_result.get("cue_results", []) if cue.get("status") != "supported"]

            lines.extend(["", "Supported contextual clues:"])
            if supported:
                for cue in supported:
                    lines.extend(format_cue_for_markdown(cue))
            else:
                lines.append("- none")

            rejected = [cue for cue in rejected_or_unclear if cue.get("status") == "rejected"]
            unclear_items = [cue for cue in rejected_or_unclear if cue.get("status") == "unclear"]
            lines.extend(["", "Rejected failure clues:"])
            if rejected:
                for cue in rejected:
                    lines.extend(format_cue_for_markdown(cue, include_evidence=False))
            else:
                lines.append("- none")

            lines.extend(["", "Unclear clues:"])
            if unclear_items:
                for cue in unclear_items:
                    lines.extend(format_cue_for_markdown(cue, include_evidence=False))
            else:
                lines.append("- none")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def participant_evidence_summary(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_shop = {
        (result.get("task", {}).get("target_shop") or ""): result
        for result in task_results
    }
    completed = [
        result.get("task", {}).get("example_task", "unknown task")
        for result in task_results
        if task_completed_from_counts(result.get("key_counts", {}))
    ]
    decathlon = by_shop.get("Decathlon", {})
    fnac = by_shop.get("Fnac", {})
    zara = by_shop.get("Zara", {})
    actual_taps = max(
        [
            int_safe(result.get("key_counts", {}).get("actual_beacon_scan_count"))
            for result in task_results
        ] or [0]
    )
    target_shop_beacon_taps = max(
        [
            int_safe(result.get("key_counts", {}).get("target_shop_beacon_scan_count"))
            for result in task_results
        ] or [0]
    )
    house_confirmed_tasks = sum(
        1 for result in task_results
        if (result.get("house_recall_evidence") or {}).get("confirmed")
    )
    house_confused_tasks = sum(
        1 for result in task_results
        if (result.get("house_recall_evidence") or {}).get("confused")
        or (
            (result.get("house_recall_evidence") or {}).get("mentioned")
            and not (result.get("house_recall_evidence") or {}).get("confirmed")
        )
    )
    return {
        "Completed tasks": len(completed),
        "Decathlon entries": count_value(decathlon, "target_shop_enter_count"),
        "Decathlon visual rows": count_value(decathlon, "target_shop_visual_attention_count"),
        "Decathlon Ask target item": count_value(decathlon, "ask_french_target_item_count"),
        "Decathlon Ask wrong item": count_value(decathlon, "ask_related_or_wrong_item_count"),
        "Decathlon Pay target item": count_value(decathlon, "pay_french_target_item_count"),
        "Decathlon Pay wrong item": count_value(decathlon, "pay_related_or_wrong_item_count"),
        "Decathlon validation": count_value(decathlon, "validate_target_intention_count"),
        "Fnac entries": count_value(fnac, "target_shop_enter_count"),
        "Fnac visual rows": count_value(fnac, "target_shop_visual_attention_count"),
        "Fnac Ask target item": count_value(fnac, "ask_french_target_item_count"),
        "Fnac Ask wrong item": count_value(fnac, "ask_related_or_wrong_item_count"),
        "Fnac Pay target item": count_value(fnac, "pay_french_target_item_count"),
        "Fnac Pay wrong item": count_value(fnac, "pay_related_or_wrong_item_count"),
        "Fnac validation": count_value(fnac, "validate_target_intention_count"),
        "Zara entries": count_value(zara, "target_shop_enter_count"),
        "Zara visual rows": count_value(zara, "target_shop_visual_attention_count"),
        "Zara Ask target item": count_value(zara, "ask_french_target_item_count"),
        "Zara Ask wrong item": count_value(zara, "ask_related_or_wrong_item_count"),
        "Zara Pay target item": count_value(zara, "pay_french_target_item_count"),
        "Zara Pay wrong item": count_value(zara, "pay_related_or_wrong_item_count"),
        "Zara validation": count_value(zara, "validate_target_intention_count"),
        "House confirmed tasks": house_confirmed_tasks,
        "House confused/unclear tasks": house_confused_tasks,
        "Actual beacon taps": actual_taps,
        "Beacon taps at target shops": target_shop_beacon_taps,
    }


def task_completed_from_counts(counts: dict[str, Any]) -> bool:
    return bool(
        int_safe(counts.get("validate_target_intention_count"))
        or (
            int_safe(counts.get("pay_french_target_item_count"))
            and int_safe(counts.get("item_in_bag_supporting_object_count"))
        )
    )


def count_value(task_result: dict[str, Any], key: str) -> int:
    return int_safe((task_result.get("key_counts") or {}).get(key))


def int_safe(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def format_cue_for_markdown(cue: dict[str, Any], include_evidence: bool = True) -> list[str]:
    lines = [
        f"- Clue {cue.get('clue_number')}: {cue.get('clue_name')}",
        f"  - Status: `{cue.get('status')}`",
        f"  - Reason: {cue.get('python_reason')}",
    ]
    evidence = cue.get("evidence") or []
    if include_evidence and evidence:
        lines.append("  - Evidence:")
        for item in evidence[:3]:
            raw_values = item.get("raw_values") or []
            if raw_values:
                lines.append(f"    - {raw_values[0]}")
            else:
                lines.append(
                    f"    - {item.get('file_name')}, rows {item.get('row_numbers')}, "
                    f"time={item.get('in_game_time') or item.get('time')}, "
                    f"verb={item.get('verb')}, object={item.get('object')}"
                )
    return lines


def enabled_event_tasks(tasks_config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        task for task in tasks_config.get("event_based_tasks", [])
        if task.get("enabled", True)
    ]


def collect_events(participant: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for env_name, environment in participant.get("environments", {}).items():
        for file_payload in environment.get("events", []):
            rows = parse_csv_payload(file_payload)
            events.extend(group_logical_events(rows, file_payload.get("file_name", ""), env_name))
    return sorted(events, key=lambda event: event.get("time_sort", 0))


def parse_csv_payload(file_payload: dict[str, Any]) -> list[dict[str, str]]:
    text = file_payload.get("raw_csv_with_row_numbers", "")
    if not text:
        return []
    return list(csv.DictReader(StringIO(text)))


def group_logical_events(rows: list[dict[str, str]], file_name: str, environment: str) -> list[dict[str, Any]]:
    in_game_times: dict[tuple[str, str, str, str, str], str] = {}
    for row in rows:
        if row.get("property") != "inGameTime":
            continue
        in_game_times[(
            row.get("time", ""),
            row.get("actor", ""),
            row.get("verb", ""),
            row.get("object", ""),
            row.get("user_id", ""),
        )] = row.get("value", "")

    events = []
    for row in rows:
        properties: dict[str, list[str]] = defaultdict(list)
        prop = row.get("property", "")
        if prop:
            properties[prop].append(row.get("value", ""))
        raw_text = ",".join(row.values())
        key = (
            row.get("time", ""),
            row.get("actor", ""),
            row.get("verb", ""),
            row.get("object", ""),
            row.get("user_id", ""),
        )
        event = {
            "file_name": file_name,
            "environment": environment,
            "rows": [row],
            "row_numbers": [row.get("row_number", "")],
            "time": row.get("time", ""),
            "time_sort": safe_int(row.get("time", "")),
            "actor": row.get("actor", ""),
            "verb": row.get("verb", ""),
            "object": row.get("object", ""),
            "properties": dict(properties),
            "shop": canonical_shop(raw_text),
            "shop_label": shop_label_from_row(row, raw_text),
            "item": canonical_item(raw_text),
            "intention": canonical_intention(raw_text),
            "text": normalize_text(raw_text),
        }
        event["in_game_time"] = first_property(event, "inGameTime") or in_game_times.get(key)
        events.append(event)
    return events


def collect_trajectory_data(participant: dict[str, Any]) -> dict[str, Any]:
    notes = []
    points = []
    for env_name, environment in participant.get("environments", {}).items():
        for file_payload in environment.get("trajectory", []):
            rows = parse_csv_payload(file_payload)
            if not rows:
                continue
            points.extend(trajectory_points_from_rows(rows, file_payload.get("file_name", ""), env_name))
            notes.append({
                "environment": env_name,
                "file_name": file_payload.get("file_name", ""),
                "row_count": len(rows),
                "first_row": row_citation(file_payload.get("file_name", ""), rows[0]),
                "last_row": row_citation(file_payload.get("file_name", ""), rows[-1]),
                "limitation": (
                    "Trajectory rows contain coordinates/rotations but no shop labels. "
                    "Python uses event timestamps plus all-participant shop events to estimate shop areas."
                ),
            })
    return {
        "available": bool(notes),
        "files": notes,
        "points": sorted(points, key=lambda point: point["time"]),
    }


def public_trajectory_summary(trajectory_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": trajectory_data.get("available", False),
        "files": trajectory_data.get("files", []),
        "point_count": len(trajectory_data.get("points", [])),
    }


def build_shop_position_model(
    raw_batch: dict[str, Any],
    shop_locations_config: dict[str, Any] | None = None
) -> dict[str, Any]:
    manual_model = shop_position_model_from_manual_locations(shop_locations_config or {})
    if manual_model.get("shops"):
        return manual_model

    samples_by_shop: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for participant in raw_batch.get("participants", []):
        participant_id = participant.get("participant_id", "")
        shopping = participant.get("environments", {}).get("ShoppingCenter", {})
        trajectory_points = []
        for trajectory_payload in shopping.get("trajectory", []):
            rows = parse_csv_payload(trajectory_payload)
            trajectory_points.extend(
                trajectory_points_from_rows(rows, trajectory_payload.get("file_name", ""), "ShoppingCenter")
            )
        trajectory_points.sort(key=lambda point: point["time"])
        if not trajectory_points:
            continue

        for events_payload in shopping.get("events", []):
            rows = parse_csv_payload(events_payload)
            for row in rows:
                if row.get("object") != "Shop" or row.get("property") != "ShopName":
                    continue
                if row.get("verb") not in {"Enter", "Exit"}:
                    continue
                canonical = canonical_shop(row.get("value", ""))
                if not canonical:
                    continue
                nearest = nearest_trajectory_point(trajectory_points, safe_int(row.get("time", "")))
                if not nearest or nearest["distance_ms"] > 2000:
                    continue
                variant = normalize_shop_variant(row.get("value", ""))
                samples_by_shop[canonical][variant].append({
                    "participant_id": participant_id,
                    "event_file": events_payload.get("file_name", ""),
                    "event_row": row.get("row_number", ""),
                    "event_time": row.get("time", ""),
                    "event_verb": row.get("verb", ""),
                    "shop_label": row.get("value", ""),
                    "trajectory_file": nearest["file_name"],
                    "trajectory_row": nearest["row_number"],
                    "trajectory_time": nearest["time"],
                    "distance_ms": nearest["distance_ms"],
                    "x": nearest["x"],
                    "y": nearest["y"],
                    "z": nearest["z"],
                })

    shops = {}
    for canonical, variants in samples_by_shop.items():
        variant_summaries = []
        for variant, samples in variants.items():
            participant_count = len({sample["participant_id"] for sample in samples})
            center = centroid(samples)
            variant_summaries.append({
                "variant": variant,
                "sample_count": len(samples),
                "participant_count": participant_count,
                "center": center,
                "typical_radius": typical_radius(samples, center),
                "example_samples": sample_citations(samples[:3]),
            })
        variant_summaries.sort(key=lambda item: (-item["participant_count"], -item["sample_count"], item["variant"]))
        representative = variant_summaries[0] if variant_summaries else None
        shops[canonical] = {
            "representative_variant": representative["variant"] if representative else None,
            "representative_center": representative["center"] if representative else None,
            "representative_radius": representative["typical_radius"] if representative else None,
            "reason": (
                "The representative variant is the shop label observed for the most participants. "
                "This handles labels like Zara, Zara1, and Zara2 that may refer to different physical positions."
            ),
            "variants": variant_summaries,
        }

    return {
        "method": (
            "For each ShoppingCenter shop Enter/Exit event row with property == ShopName, Python found the nearest "
            "trajectory row for the same participant. Shop centers are averages of those matched HMD positions "
            "across all loaded participants. This is an inferred VR shop-area model, not a manually provided map."
        ),
        "nearest_trajectory_window_ms": 2000,
        "shops": shops,
    }


def shop_position_model_from_manual_locations(shop_locations_config: dict[str, Any]) -> dict[str, Any]:
    entrances_by_shop = shop_locations_config.get("shop_entrances", {}) if isinstance(shop_locations_config, dict) else {}
    shops = {}
    for shop_name, entrances in entrances_by_shop.items():
        canonical = canonical_shop(shop_name) or shop_name
        normalized_entrances = []
        for index, entrance in enumerate(entrances if isinstance(entrances, list) else []):
            x = safe_float(entrance.get("x"))
            y = safe_float(entrance.get("y"))
            z = safe_float(entrance.get("z"))
            if x is None or y is None or z is None:
                continue
            normalized_entrances.append({
                "label": f"{canonical} entrance {index + 1}",
                "x": x,
                "y": y,
                "z": z,
                "floor": floor_from_y(y),
            })
        if not normalized_entrances:
            continue
        center = centroid(normalized_entrances)
        shops[canonical] = {
            "representative_variant": canonical,
            "representative_center": center,
            "representative_radius": 3.0,
            "entrance_radius": 3.0,
            "entrances": normalized_entrances,
            "reason": "Manual store entrance coordinates provided by the analyst.",
            "variants": [
                {
                    "variant": canonical,
                    "sample_count": len(normalized_entrances),
                    "participant_count": None,
                    "center": center,
                    "typical_radius": 3.0,
                    "example_samples": normalized_entrances,
                }
            ],
        }
    return {
        "method": (
            "Manual store entrance coordinates provided by the analyst. "
            "These coordinates support trajectory/proximity reasoning only; event-log Verb == Enter/Exit remains the source for actual shop entry/exit."
        ),
        "nearest_trajectory_window_ms": None,
        "shops": shops,
        "source": "manual_shop_entrance_coordinates",
        "coordinate_system": shop_locations_config.get("coordinate_system", {}),
        "important_note": shop_locations_config.get("important_note", ""),
    }


def trajectory_points_from_rows(rows: list[dict[str, str]], file_name: str, environment: str) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        time_value = safe_int(row.get("UnixTime", "") or row.get("time", ""))
        x = safe_float(row.get("HMDPosX"))
        y = safe_float(row.get("HMDPosY"))
        z = safe_float(row.get("HMDPosZ"))
        if time_value <= 0 or x is None or y is None or z is None:
            continue
        points.append({
            "file_name": file_name,
            "environment": environment,
            "row_number": row.get("row_number", ""),
            "time": time_value,
            "x": x,
            "y": y,
            "z": z,
            "floor": floor_from_y(y),
        })
    return points


def nearest_trajectory_point(points: list[dict[str, Any]], event_time: int) -> dict[str, Any] | None:
    if not points or event_time <= 0:
        return None
    best = min(points, key=lambda point: abs(point["time"] - event_time))
    return {
        **best,
        "distance_ms": abs(best["time"] - event_time),
    }


def trajectory_task_evidence(
    trajectory_data: dict[str, Any],
    shop_position_model: dict[str, Any],
    target_shop: str
) -> dict[str, Any]:
    points = trajectory_data.get("points", [])
    shop_model = shop_position_model.get("shops", {}).get(target_shop)
    if not points:
        return {
            "status": "unavailable",
            "reason": "No trajectory points were loaded for this participant/scenario.",
        }
    if not shop_model or not shop_model.get("representative_center"):
        return {
            "status": "unavailable",
            "reason": f"No inferred shop-position model is available for {target_shop}.",
        }

    center = shop_model["representative_center"]
    target_floor = floor_from_y(float(center.get("y", 0.0)))
    radius = max(2.0, float(shop_model.get("representative_radius") or 0.0))
    near_radius = radius + 2.0
    nearest = min(points, key=lambda point: distance_3d(point, center))
    nearest_distance = distance_3d(nearest, center)
    near_points = [point for point in points if distance_3d(point, center) <= near_radius]
    time_near_seconds = estimate_time_near_seconds(near_points)
    status = "near_inferred_target_shop_area" if nearest_distance <= near_radius else "not_near_inferred_target_shop_area"

    return {
        "status": status,
        "target_shop": target_shop,
        "representative_shop_variant": shop_model.get("representative_variant"),
        "inferred_shop_center": center,
        "inferred_target_floor": target_floor,
        "near_radius_used": round(near_radius, 3),
        "nearest_distance": round(nearest_distance, 3),
        "nearest_trajectory_point": trajectory_point_citation(nearest),
        "nearest_floor": nearest.get("floor"),
        "floors_seen_in_shoppingcenter": sorted({point.get("floor") for point in points if point.get("floor")}),
        "floors_seen_near_shop": sorted({point.get("floor") for point in near_points if point.get("floor")}),
        "near_point_count": len(near_points),
        "estimated_time_near_seconds": round(time_near_seconds, 3),
        "first_near_point": trajectory_point_citation(near_points[0]) if near_points else None,
        "last_near_point": trajectory_point_citation(near_points[-1]) if near_points else None,
        "caution": (
            "This is inferred from cross-participant shop-event/trajectory alignment. "
            "It can support movement reasoning, but Verb == Enter/Exit remains the source for actual shop entry/exit."
        ),
    }


def house_recall_evidence(events: list[dict[str, Any]], target_shop: str, target_item: str) -> dict[str, Any]:
    house_events = [
        event for event in events
        if event.get("environment") == "House"
        and normalize_text(event.get("verb", "")) in {"say", "confirm", "confuse"}
        and normalize_text(event.get("object", "")) in {"phrase", "intention"}
    ]
    if not house_events:
        return {
            "house_logs_available": any(event.get("environment") == "House" for event in events),
            "mentioned": False,
            "confirmed": False,
            "confused": False,
            "mentioned_events": [],
            "confirmed_events": [],
            "confused_events": [],
        }

    grouped = group_house_recall_events(house_events)
    mentioned = []
    confirmed = []
    confused = []
    for group in grouped:
        text = group.get("text", "")
        matches_item = canonical_item(text) == target_item
        matches_shop = canonical_shop(text) == target_shop
        if not (matches_item or matches_shop):
            continue
        if matches_item:
            mentioned.append(group)
        normalized_verb = normalize_text(group.get("verb", ""))
        if matches_item and matches_shop and normalized_verb == "confirm":
            confirmed.append(group)
        if matches_item and normalized_verb == "confuse":
            confused.append(group)

    return {
        "house_logs_available": True,
        "mentioned": bool(mentioned),
        "confirmed": bool(confirmed),
        "confused": bool(confused),
        "mentioned_events": mentioned[:6],
        "confirmed_events": confirmed[:6],
        "confused_events": confused[:6],
    }


def group_house_recall_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for event in events:
        key = (
            event.get("file_name", ""),
            event.get("time", ""),
            event.get("verb", ""),
            event.get("object", ""),
        )
        group = groups.setdefault(key, {
            "file_name": event.get("file_name"),
            "environment": event.get("environment"),
            "rows": [],
            "row_numbers": [],
            "time": event.get("time", ""),
            "time_sort": event.get("time_sort", 0),
            "actor": event.get("actor"),
            "verb": event.get("verb"),
            "object": event.get("object"),
            "in_game_time": event.get("in_game_time"),
            "shop": None,
            "item": None,
            "intention": None,
            "text": "",
        })
        group["rows"].extend(event.get("rows", []))
        group["row_numbers"].extend(event.get("row_numbers", []))
        if not group.get("in_game_time") and event.get("in_game_time"):
            group["in_game_time"] = event.get("in_game_time")

    output = []
    for group in groups.values():
        raw_text = " ".join(
            " ".join(str(value) for value in row.values() if value)
            for row in group["rows"]
        )
        group["text"] = normalize_text(raw_text)
        group["shop"] = canonical_shop(raw_text)
        group["item"] = canonical_item(raw_text)
        group["intention"] = canonical_intention(raw_text)
        output.append(group)
    return sorted(output, key=lambda item: item.get("time_sort", 0))


def centroid(samples: list[dict[str, Any]]) -> dict[str, float]:
    count = max(1, len(samples))
    return {
        "x": round(sum(sample["x"] for sample in samples) / count, 3),
        "y": round(sum(sample["y"] for sample in samples) / count, 3),
        "z": round(sum(sample["z"] for sample in samples) / count, 3),
    }


def typical_radius(samples: list[dict[str, Any]], center: dict[str, float]) -> float:
    if not samples:
        return 2.0
    distances = sorted(distance_3d(sample, center) for sample in samples)
    index = min(len(distances) - 1, int(len(distances) * 0.75))
    return round(max(2.0, distances[index]), 3)


def distance_3d(point: dict[str, Any], center: dict[str, float]) -> float:
    return math.sqrt(
        (float(point["x"]) - float(center["x"])) ** 2
        + (float(point["y"]) - float(center["y"])) ** 2
        + (float(point["z"]) - float(center["z"])) ** 2
    )


def estimate_time_near_seconds(points: list[dict[str, Any]]) -> float:
    if len(points) < 2:
        return 0.0
    total_ms = 0
    previous = points[0]
    for point in points[1:]:
        delta = point["time"] - previous["time"]
        if 0 < delta <= 2000:
            total_ms += delta
        previous = point
    return total_ms / 1000


def sample_citations(samples: list[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{sample['participant_id']}: {sample['event_file']}, row {sample['event_row']}, "
            f"verb={sample['event_verb']}, ShopName={sample['shop_label']} matched to "
            f"{sample['trajectory_file']}, row {sample['trajectory_row']}, "
            f"position=({sample['x']:.3f}, {sample['y']:.3f}, {sample['z']:.3f})"
        )
        for sample in samples
    ]


def trajectory_point_citation(point: dict[str, Any]) -> dict[str, Any]:
    return {
        "file_name": point.get("file_name"),
        "row_number": point.get("row_number"),
        "time": point.get("time"),
        "position": {
            "x": round(float(point.get("x", 0.0)), 3),
            "y": round(float(point.get("y", 0.0)), 3),
            "z": round(float(point.get("z", 0.0)), 3),
        },
        "floor": point.get("floor"),
    }


def floor_from_y(y: float) -> str:
    if y < 2:
        return "floor0"
    if 6.5 < y < 8.5:
        return "floor1"
    if 12.5 < y < 14.5:
        return "floor2"
    return "unclear"


def analyze_task(
    events: list[dict[str, Any]],
    trajectory_summary: dict[str, Any],
    task: dict[str, Any],
    clues_config: dict[str, Any],
    shop_position_model: dict[str, Any]
) -> dict[str, Any]:
    target_shop = task.get("target_shop", "")
    target_item = task.get("target_item", "")
    target_shop_events = [event for event in events if event.get("shop") == target_shop]
    entered = [event for event in target_shop_events if is_verb(event, "Enter")]
    exited = [event for event in target_shop_events if is_verb(event, "Exit")]
    watched = [event for event in target_shop_events if event.get("verb", "").lower() in WATCH_VERBS]
    buying_actions = [event for event in events if event_matches_task_buying(event, target_shop, target_item)]
    payments = [event for event in events if is_payment(event, target_item)]
    bag_events = [event for event in events if is_bag_event(event, target_item)]
    validations = [event for event in events if is_validation(event, target_item)]
    npc_or_ask = [event for event in events if is_npc_or_ask(event, target_shop, target_item)]
    npc_interactions = [event for event in target_shop_events if is_npc_interaction(event)]
    ask_target_item = [event for event in events if is_ask_strict_target_item(event, target_item)]
    ask_related_or_wrong_item = [event for event in events if is_ask_related_or_wrong_item(event, target_item)]
    pay_related_or_wrong_item = [event for event in events if is_pay_related_or_wrong_item(event, target_item)]
    secondary = [event for event in events if event.get("environment") == "ShoppingCenter" and is_secondary_event(event)]
    target_shop_secondary = [
        event for event in secondary
        if canonical_shop(event.get("text", "")) in {"Decathlon", "Fnac", "Zara"}
    ]
    wristwatch = [
        event for event in events
        if event.get("environment") == "ShoppingCenter"
        and "wristwatch" in event.get("object", "").lower()
    ]
    phone = [
        event for event in events
        if event.get("environment") == "ShoppingCenter"
        and ("phone" in event.get("object", "").lower() or "phone" in event.get("text", ""))
    ]
    forgot = [
        event for event in events
        if event.get("environment") == "ShoppingCenter"
        and "forgot" in event.get("text", "")
    ]
    house_recall = house_recall_evidence(events, target_shop, target_item)
    irrelevant_shop_events = [
        event for event in events
        if event.get("shop")
        and event.get("shop") != target_shop
        and is_verb(event, "Enter")
    ]

    completed = bool(validations or (payments and bag_events))
    completion_evidence = completion_evidence_by_step(
        entered=entered,
        npc_or_ask=npc_or_ask,
        payments=payments,
        bag_events=bag_events,
        validations=validations,
    )
    trajectory_evidence = trajectory_task_evidence(
        trajectory_summary,
        shop_position_model,
        target_shop
    )
    cue_results = []
    for clue in clues_config.get("clues", []):
        number = clue.get("number")
        cue_results.append(calculate_clue(
            number=number,
            clue=clue,
            target_shop=target_shop,
            target_item=target_item,
            completed=completed,
            entered=entered,
            exited=exited,
            watched=watched,
            buying_actions=buying_actions,
            payments=payments,
            bag_events=bag_events,
            validations=validations,
            npc_or_ask=npc_or_ask,
            secondary=secondary,
            wristwatch=wristwatch,
            phone=phone,
            forgot=forgot,
            house_recall=house_recall,
            irrelevant_shop_events=irrelevant_shop_events,
            trajectory_summary=trajectory_summary,
            trajectory_evidence=trajectory_evidence,
        ))

    return {
        "task": task,
        "python_task_status": task_status(completed, payments, bag_events, validations, entered, watched, buying_actions),
        "key_counts": {
            "target_shop_visual_attention_count": len(watched),
            "target_shop_enter_count": len(entered),
            "target_shop_exit_count": len(exited),
            "target_npc_interaction_count": len(npc_interactions),
            "ask_french_target_item_count": len(ask_target_item),
            "ask_related_or_wrong_item_count": len(ask_related_or_wrong_item),
            "pay_french_target_item_count": len(payments),
            "pay_related_or_wrong_item_count": len(pay_related_or_wrong_item),
            "item_in_bag_supporting_object_count": len(bag_events),
            "validate_target_intention_count": len(validations),
            "raw_scan_tag_row_count": len(secondary),
            "actual_beacon_scan_count": unique_event_time_count(secondary),
            "target_shop_beacon_scan_count": unique_event_time_count(target_shop_secondary),
        },
        "numeric_evidence": numeric_evidence_table(
            house_recall=house_recall,
            watched=watched,
            entered=entered,
            exited=exited,
            npc_interactions=npc_interactions,
            ask_target_item=ask_target_item,
            ask_related_or_wrong_item=ask_related_or_wrong_item,
            payments=payments,
            pay_related_or_wrong_item=pay_related_or_wrong_item,
            bag_events=bag_events,
            validations=validations,
            secondary=secondary,
            phone=phone,
            wristwatch=wristwatch,
            trajectory_evidence=trajectory_evidence,
        ),
        "completion_evidence": completion_evidence,
        "cue_results": cue_results,
        "house_recall_evidence": house_recall,
        "trajectory_evidence": trajectory_evidence,
        "trajectory_note": trajectory_note(trajectory_summary, trajectory_evidence),
    }


def calculate_clue(**context: Any) -> dict[str, Any]:
    number = context["number"]
    clue = context["clue"]
    target_shop = context["target_shop"]
    target_item = context["target_item"]
    completed = context["completed"]
    entered = context["entered"]
    watched = context["watched"]
    buying_actions = context["buying_actions"]
    payments = context["payments"]
    bag_events = context["bag_events"]
    validations = context["validations"]
    npc_or_ask = context["npc_or_ask"]
    secondary = context["secondary"]
    wristwatch = context["wristwatch"]
    phone = context["phone"]
    forgot = context["forgot"]
    house_recall = context["house_recall"]
    irrelevant_shop_events = context["irrelevant_shop_events"]
    trajectory_evidence = context.get("trajectory_evidence", {})

    if number == 1:
        if entered and not buying_actions:
            return supported(clue, "Participant entered the target shop but Python found no Ask/Pay/ItemInBag/Validate evidence.", entered)
        return rejected(clue, "Not supported: either no Enter event exists for the target shop, or buying-related evidence exists.", entered[:2] + buying_actions[:2])

    if number == 2:
        if not entered:
            return supported(clue, f"No Verb == Enter event was found for {target_shop}. StopToWatch/Watched events were not counted as entry.", watched[:4])
        return rejected(clue, f"At least one Verb == Enter event was found for {target_shop}.", entered[:4])

    if number == 3:
        if trajectory_evidence.get("status") == "unavailable":
            if completed:
                return rejected(clue, "The task has completion evidence, so the participant reached the relevant shopping context, but trajectory floor evidence was unavailable.", [])
            return unclear(clue, trajectory_evidence.get("reason", "No trajectory evidence was available for floor reasoning."), [])
        target_floor = trajectory_evidence.get("inferred_target_floor")
        floors_seen = set(trajectory_evidence.get("floors_seen_in_shoppingcenter") or [])
        floors_near_shop = set(trajectory_evidence.get("floors_seen_near_shop") or [])
        if not target_floor or target_floor == "unclear":
            if completed:
                return rejected(clue, "The task has completion evidence, but the inferred target-shop floor is unclear from the shop-position model.", [])
            return unclear(clue, "The inferred target-shop floor is unclear from the shop-position model.", [])
        if (
            trajectory_evidence.get("status") == "near_inferred_target_shop_area"
            or target_floor in floors_near_shop
        ):
            return rejected(
                clue,
                f"Trajectory evidence places the participant near the inferred {target_shop} area on {target_floor}.",
                [trajectory_evidence.get("nearest_trajectory_point")] if trajectory_evidence.get("nearest_trajectory_point") else []
            )
        if target_floor in floors_seen:
            return rejected(
                clue,
                f"Trajectory evidence shows the participant visited {target_floor}, the inferred floor for {target_shop}, even though Python did not infer proximity to the shop area.",
                [trajectory_evidence.get("nearest_trajectory_point")] if trajectory_evidence.get("nearest_trajectory_point") else []
            )
        if floors_seen:
            return supported(
                clue,
                f"Trajectory evidence did not show the participant on {target_floor}, the inferred floor for {target_shop}. Floors seen: {', '.join(sorted(floors_seen))}.",
                [trajectory_evidence.get("nearest_trajectory_point")] if trajectory_evidence.get("nearest_trajectory_point") else []
            )
        return unclear(clue, "Trajectory data was loaded, but Python could not identify any clear floor visits.", [])

    if number == 4:
        if npc_or_ask and not payments:
            return supported(clue, "NPC/Ask-like evidence exists, but no payment evidence for the target item was found.", npc_or_ask[:4])
        if payments:
            return rejected(clue, "Payment evidence for the target item exists.", payments[:4])
        return unclear(clue, "No NPC/Ask-like evidence was found for the target shop/item.", [])

    if number == 5:
        if payments and not bag_events:
            return supported(clue, "Payment evidence exists, but no target item bag-placement evidence was found.", payments[:4])
        if payments and bag_events:
            return rejected(clue, "Payment and target item bag-placement evidence both exist.", payments[:2] + bag_events[:2])
        return unclear(clue, "No payment evidence for the target item was found.", [])

    if number == 6:
        if len(entered) > 1 and not completed:
            return supported(clue, "Multiple Enter events for the target shop exist, but completion evidence was not found.", entered[:4])
        return rejected(clue, "Python did not find multiple target-shop Enter events without completion.", entered[:4])

    if number == 7:
        if completed:
            return rejected(clue, "The task has completion evidence, so this missed-task irrelevant-shop clue is not applicable.", [])
        if not completed and irrelevant_shop_events:
            return supported(clue, "The task was not completed and Python found real Verb == Enter evidence for other shops. Watch/Watched rows are not counted for this clue.", irrelevant_shop_events[:6])
        return unclear(clue, "No clear irrelevant-shop entry was found, or the task appears completed. Watch/Watched rows are not counted as spending time inside another shop.", irrelevant_shop_events[:4])

    if number == 8:
        if completed:
            return rejected(clue, "The task has completion evidence, so this failed-task secondary-distraction clue is not applicable.", [])
        first_attempt = first_task_attempt_event(
            entered=entered,
            npc_or_ask=npc_or_ask,
            payments=payments,
            validations=validations,
            buying_actions=buying_actions,
        )
        first_attempt_time = first_attempt.get("time_sort") if first_attempt else None
        if first_attempt_time and any(event["time_sort"] < first_attempt_time for event in secondary):
            nearest_secondary = max(
                [event for event in secondary if event["time_sort"] < first_attempt_time],
                key=lambda event: event["time_sort"]
            )
            gap_seconds = (first_attempt_time - nearest_secondary["time_sort"]) / 1000
            return supported(
                clue,
                f"A Tag/Scan secondary event happened {gap_seconds:.1f} seconds before the first target-shop attempt. Python treats this as candidate sequence evidence without one exact time limit; the reasoner should judge behavioral immediacy from the surrounding sequence.",
                [nearest_secondary, first_attempt]
            )
        first_search_context = first_sorted_event(watched)
        first_search_time = first_search_context.get("time_sort") if first_search_context else None
        if first_search_time and any(event["time_sort"] < first_search_time for event in secondary):
            nearest_secondary = max(
                [event for event in secondary if event["time_sort"] < first_search_time],
                key=lambda event: event["time_sort"]
            )
            gap_seconds = (first_search_time - nearest_secondary["time_sort"]) / 1000
            return unclear(
                clue,
                f"A Tag/Scan secondary event happened {gap_seconds:.1f} seconds before the first target-shop visual-attention context. Because Watch/StartToWatch is not a true task attempt, Python marks this as candidate distraction evidence for the reasoner rather than direct support.",
                [nearest_secondary, first_search_context]
            )
        if secondary:
            return unclear(clue, "Secondary Tag/Scan events exist, but Python did not find a clear target-shop attempt or visual-attention context to compare the sequence against.", secondary[:4])
        return rejected(clue, "No secondary Tag/Scan event was found.", [])

    if number == 9:
        if completed:
            return rejected(clue, "The task has completion evidence, so target-shop-only-for-secondary is not applicable.", [])
        secondary_inside = events_inside_target_shop(secondary, entered, context["exited"])
        if entered and secondary_inside and not buying_actions:
            return supported(clue, "A Tag/Scan event occurred during a target-shop Enter/Exit interval and no buying action was found.", secondary_inside[:4])
        return unclear(clue, "Python did not find enough evidence that target-shop entry was only for secondary task.", secondary_inside[:4])

    if number == 10:
        task_context_events = sorted_events(entered + watched + npc_or_ask + payments + validations + buying_actions)
        contextual_wristwatch = events_between_task_context(
            wristwatch,
            task_context_events=task_context_events,
            completion_event=first_completion_event(validations=validations, bag_events=bag_events, payments=payments)
        )
        watch_episode_count = distinct_event_episode_count(contextual_wristwatch)
        global_watch_episode_count = distinct_event_episode_count(wristwatch)
        if completed:
            if not wristwatch:
                return rejected(
                    clue,
                    "The task has completion evidence and Python found no ShoppingCenter Wristwatch evidence.",
                    []
                )
            if not contextual_wristwatch:
                return unclear(
                    clue,
                    f"Python found {global_watch_episode_count} ShoppingCenter Wristwatch episodes, but none were linked to the target-shop attempt/search sequence.",
                    wristwatch[:6]
                )
            if watch_episode_count < 2:
                return rejected(
                    clue,
                    f"Only {watch_episode_count} task-context Wristwatch episode was found. Watch distraction requires repeated or sustained watch-looking in relation to the task sequence.",
                    contextual_wristwatch[:6]
                )
            return unclear(
                clue,
                f"Python found {watch_episode_count} task-context Wristwatch episodes. For a completed task this is contextual only; the reasoner should judge whether repeated watch-looking suggests distraction from the cited rows.",
                contextual_wristwatch[:6]
            )
        if watch_episode_count >= 2:
            return unclear(
                clue,
                f"Python found {watch_episode_count} task-context Wristwatch episodes. The reasoner should judge whether repeated watch-looking suggests distraction from the cited rows and task sequence.",
                contextual_wristwatch[:6]
            )
        if contextual_wristwatch:
            return rejected(
                clue,
                f"Only {watch_episode_count} task-context Wristwatch episode was found. Watch distraction requires repeated or sustained watch-looking in relation to the task sequence.",
                contextual_wristwatch[:6]
            )
        if wristwatch:
            return unclear(
                clue,
                f"Python found {global_watch_episode_count} ShoppingCenter Wristwatch episodes, but none were clearly linked to the target-shop attempt/search sequence.",
                wristwatch[:6]
            )
        return rejected(clue, "No ShoppingCenter Wristwatch event was found.", [])

    if number == 11:
        if completed:
            return rejected(clue, "The task has completion evidence, so phone-use-as-failure clue is not applicable.", [])
        phone_inside = events_inside_target_shop(phone, entered, context["exited"])
        if phone_inside and not buying_actions:
            return supported(clue, "Phone-related events occurred during a target-shop Enter/Exit interval, with no buying action found.", phone_inside[:4])
        return unclear(clue, "Python did not find enough evidence that phone use distracted the participant near/inside the target shop.", phone[:4])

    if number == 12:
        forgot_inside = events_inside_target_shop(forgot, entered, context["exited"])
        target_forgot = [event for event in forgot if target_shop.lower() in event.get("text", "") or target_item.lower() in event.get("text", "")]
        if forgot_inside or target_forgot:
            return supported(clue, "Forgot-intention evidence was found for the target shop/item context.", (forgot_inside + target_forgot)[:4])
        return rejected(clue, "No forgot-intention event was found for the target shop/item context.", forgot[:4])

    if number == 13:
        if not house_recall.get("house_logs_available"):
            return unclear(clue, "House logs were not included in the Python clue input.", [])
        if house_recall.get("confirmed"):
            return rejected(clue, "House logs contain clear Confirm Intention evidence for this task: Verb == Confirm and Object == Intention rows have property values matching the target item and target shop.", house_recall.get("confirmed_events", []))
        if house_recall.get("confused"):
            return supported(clue, "House logs contain Confuse Intention evidence for this task and no clear Confirm Intention evidence was found. This means recall was confused or not clearly confirmed, not necessarily absent.", house_recall.get("confused_events", []))
        if not house_recall.get("mentioned"):
            return supported(clue, "House logs were included, but Python found no Say, Confirm, or Confuse evidence matching this task.", [])
        return unclear(clue, "The task was mentioned in House, but Python did not find a clear Confirm Intention row for it. Treat this as partial/unclear recall, not absence of memory.", house_recall.get("mentioned_events", []))

    if number == 14:
        first_attempt = first_task_attempt_event(
            entered=entered,
            npc_or_ask=npc_or_ask,
            payments=payments,
            validations=validations,
            buying_actions=buying_actions,
        )
        completion_event = first_completion_event(validations=validations, bag_events=bag_events, payments=payments)
        ask_event = first_matching_event(npc_or_ask, lambda event: "ask" in event.get("text", ""))
        late_purchase_event = first_completion_event(validations=validations, bag_events=bag_events, payments=payments)
        returned_before_completion = participant_left_and_returned_before_completion(entered, context["exited"], completion_event)
        if first_attempt and completion_event:
            delay_seconds = (completion_event["time_sort"] - first_attempt["time_sort"]) / 1000
            ask_gap_seconds = None
            if ask_event and late_purchase_event and late_purchase_event["time_sort"] > ask_event["time_sort"]:
                ask_gap_seconds = (late_purchase_event["time_sort"] - ask_event["time_sort"]) / 1000
            if returned_before_completion:
                return supported(
                    clue,
                    "The participant left the target shop and returned before completing the task.",
                    entered[:4] + context["exited"][:4] + [completion_event]
                )
            if ask_gap_seconds is not None and ask_gap_seconds > DELAY_THRESHOLD_SECONDS:
                return supported(
                    clue,
                    f"The gap between Ask and completion evidence was {ask_gap_seconds:.1f} seconds, greater than the {DELAY_THRESHOLD_SECONDS}-second delay threshold.",
                    [ask_event, late_purchase_event]
                )
            if delay_seconds > DELAY_THRESHOLD_SECONDS:
                return unclear(
                    clue,
                    f"The gap from first target-shop attempt to completion was {delay_seconds:.1f} seconds. Python treats this as candidate delay evidence for the reasoner, but does not support delay from elapsed time alone.",
                    [first_attempt, completion_event]
                )
            if completion_event["time_sort"] >= first_attempt["time_sort"]:
                return rejected(
                    clue,
                    f"The task was completed in the same short shopping sequence; first attempt to completion was {delay_seconds:.1f} seconds and Python found no leave-and-return or large Ask-to-completion gap.",
                    [first_attempt, completion_event]
                )
        if completed:
            return unclear(clue, "Completion exists, but Python could not establish a reliable first-attempt-to-completion timing window.", [completion_event] if completion_event else [])
        return rejected(clue, "No completion evidence was found.", [])

    return unclear(clue, "This clue number is not implemented in the Python clue calculator.", [])


def completion_evidence_by_step(
    *,
    entered: list[dict[str, Any]],
    npc_or_ask: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    bag_events: list[dict[str, Any]],
    validations: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    ask_events = [event for event in npc_or_ask if "ask" in event.get("text", "")]
    return {
        "Enter target shop": sorted(entered, key=lambda event: event.get("time_sort", 0))[:3],
        "Ask target item": sorted(ask_events or npc_or_ask, key=lambda event: event.get("time_sort", 0))[:3],
        "Pay target item": sorted(payments, key=lambda event: event.get("time_sort", 0))[:3],
        "Place ItemInBag for target item or alias": sorted(bag_events, key=lambda event: event.get("time_sort", 0))[:3],
        "Validate target intention": sorted(validations, key=lambda event: event.get("time_sort", 0))[:3],
    }


def numeric_evidence_table(
    *,
    house_recall: dict[str, Any],
    watched: list[dict[str, Any]],
    entered: list[dict[str, Any]],
    exited: list[dict[str, Any]],
    npc_interactions: list[dict[str, Any]],
    ask_target_item: list[dict[str, Any]],
    ask_related_or_wrong_item: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    pay_related_or_wrong_item: list[dict[str, Any]],
    bag_events: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    phone: list[dict[str, Any]],
    wristwatch: list[dict[str, Any]],
    trajectory_evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "house_confirm_count": {
            "label": "House Confirm rows",
            "value": len(house_recall.get("confirmed_events") or []),
            "meaning": "Rows showing House confirmation of the target task.",
        },
        "house_confuse_count": {
            "label": "House Confuse rows",
            "value": len(house_recall.get("confused_events") or []),
            "meaning": "Rows showing possible House confusion about the target task.",
        },
        "target_shop_visual_attention_count": {
            "label": "Target-shop visual-attention rows",
            "value": len(watched),
            "meaning": "Watch/Watched/StartToWatch/StopToWatch rows for the target shop; looking only, not entry.",
        },
        "target_shop_enter_count": {
            "label": "Strict Enter target-shop rows",
            "value": len(entered),
            "meaning": "Only Verb == Enter and Object == Shop counts as shop entry.",
        },
        "target_shop_exit_count": {
            "label": "Strict Exit target-shop rows",
            "value": len(exited),
            "meaning": "Only Verb == Exit and Object == Shop counts as shop exit.",
        },
        "target_npc_interaction_count": {
            "label": "NPC interaction rows in target shop",
            "value": len(npc_interactions),
            "meaning": "Rows showing interaction with an NPC while in the target-shop context.",
        },
        "ask_french_target_item_count": {
            "label": "Ask French target item rows",
            "value": len(ask_target_item),
            "meaning": "Ask rows matching the French target item, which is the source of truth.",
        },
        "ask_related_or_wrong_item_count": {
            "label": "Ask related/wrong item rows",
            "value": len(ask_related_or_wrong_item),
            "meaning": "Ask rows for related or non-target items, useful for wrong-item or incomplete execution.",
        },
        "pay_french_target_item_count": {
            "label": "Pay French target item rows",
            "value": len(payments),
            "meaning": "Pay rows matching the French target item.",
        },
        "pay_related_or_wrong_item_count": {
            "label": "Pay related/wrong item rows",
            "value": len(pay_related_or_wrong_item),
            "meaning": "Pay rows for related or non-target items.",
        },
        "item_in_bag_supporting_object_count": {
            "label": "ItemInBag supporting object rows",
            "value": len(bag_events),
            "meaning": "Bag-placement rows for object/clone representations. These support but do not alone prove target completion.",
        },
        "validate_target_intention_count": {
            "label": "Validate target intention rows",
            "value": len(validations),
            "meaning": "Validate rows matching the target intention code.",
        },
        "raw_scan_tag_row_count": {
            "label": "Raw Scan/Tag rows",
            "value": len(secondary),
            "meaning": "Raw rows where Verb == Scan and Object == Tag.",
        },
        "actual_beacon_scan_count": {
            "label": "Actual beacon taps",
            "value": unique_event_time_count(secondary),
            "meaning": "Unique beacon taps after grouping duplicate Scan/Tag property rows by timestamp.",
        },
        "phone_context_count": {
            "label": "Phone context rows",
            "value": len(phone),
            "meaning": "ShoppingCenter phone rows. They are contextual unless linked to the task sequence.",
        },
        "wristwatch_context_count": {
            "label": "Wristwatch context rows",
            "value": len(wristwatch),
            "meaning": "ShoppingCenter wristwatch rows. Repeated task-linked rows may suggest distraction.",
        },
        "trajectory_nearest_distance": {
            "label": "Trajectory nearest distance",
            "value": trajectory_evidence.get("nearest_distance", "not counted"),
            "meaning": "Nearest distance to the inferred or manual target-shop entrance.",
        },
        "trajectory_time_near_seconds": {
            "label": "Trajectory time near entrance",
            "value": trajectory_evidence.get("estimated_time_near_seconds", "not counted"),
            "meaning": "Estimated time near the target-shop entrance.",
        },
    }


def first_task_attempt_event(
    *,
    entered: list[dict[str, Any]],
    npc_or_ask: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    buying_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return first_sorted_event(entered + npc_or_ask + payments + validations + buying_actions)


def first_completion_event(
    *,
    validations: list[dict[str, Any]],
    bag_events: list[dict[str, Any]],
    payments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return first_sorted_event(validations + bag_events + payments)


def first_sorted_event(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    real_events = [event for event in events if event and event.get("time_sort", 0) > 0]
    if not real_events:
        return None
    return min(real_events, key=lambda event: event.get("time_sort", 0))


def sorted_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [event for event in events if event and event.get("time_sort", 0) > 0],
        key=lambda event: event.get("time_sort", 0)
    )


def events_between_task_context(
    events: list[dict[str, Any]],
    *,
    task_context_events: list[dict[str, Any]],
    completion_event: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not events or not task_context_events:
        return []
    start = task_context_events[0].get("time_sort", 0)
    end_candidates = [event.get("time_sort", 0) for event in task_context_events]
    if completion_event:
        end_candidates.append(completion_event.get("time_sort", 0))
    end = max(end_candidates)
    return [
        event for event in sorted_events(events)
        if start <= event.get("time_sort", 0) <= end
    ]


def first_matching_event(events: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    return first_sorted_event([event for event in events if predicate(event)])


def distinct_event_episode_count(events: list[dict[str, Any]]) -> int:
    return len({
        (
            event.get("file_name", ""),
            event.get("time", ""),
            event.get("actor", ""),
            event.get("object", ""),
        )
        for event in events
    })


def participant_left_and_returned_before_completion(
    entered: list[dict[str, Any]],
    exited: list[dict[str, Any]],
    completion_event: dict[str, Any] | None,
) -> bool:
    if not completion_event or len(entered) < 2:
        return False
    completion_time = completion_event.get("time_sort", 0)
    exits_before_completion = [
        event for event in exited
        if 0 < event.get("time_sort", 0) < completion_time
    ]
    returns_before_completion = [
        event for event in entered
        if 0 < event.get("time_sort", 0) < completion_time
    ]
    return bool(exits_before_completion and len(returns_before_completion) > 1)


def task_status(
    completed: bool,
    payments: list[dict[str, Any]],
    bag_events: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    entered: list[dict[str, Any]],
    watched: list[dict[str, Any]],
    buying_actions: list[dict[str, Any]],
) -> str:
    if completed:
        return "completed_evidence_found"
    if payments and not bag_events:
        return "paid_but_no_bag_evidence"
    if entered and not buying_actions:
        return "entered_shop_no_buying_action_evidence"
    if not entered and watched:
        return "watched_or_stopped_at_shop_but_no_enter_evidence"
    if not entered:
        return "no_target_shop_enter_evidence"
    return "unclear"


def supported(clue: dict[str, Any], reason: str, evidence_events: list[dict[str, Any]]) -> dict[str, Any]:
    return clue_result(clue, "supported", reason, evidence_events)


def rejected(clue: dict[str, Any], reason: str, evidence_events: list[dict[str, Any]]) -> dict[str, Any]:
    return clue_result(clue, "rejected", reason, evidence_events)


def unclear(clue: dict[str, Any], reason: str, evidence_events: list[dict[str, Any]]) -> dict[str, Any]:
    return clue_result(clue, "unclear", reason, evidence_events)


def clue_result(clue: dict[str, Any], status: str, reason: str, evidence_events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clue_number": clue.get("number"),
        "clue_name": clue.get("name"),
        "status": status,
        "python_reason": reason,
        "evidence": [event_citation(event) for event in evidence_events[:6]],
    }


def event_citation(event: dict[str, Any]) -> dict[str, Any]:
    if "position" in event and "row_number" in event:
        position = event.get("position") or {}
        return {
            "file_name": event.get("file_name"),
            "row_numbers": [event.get("row_number")],
            "time": event.get("time"),
            "in_game_time": None,
            "verb": "trajectory",
            "object": "HMDPosition",
            "shop": None,
            "item": None,
            "raw_values": [
                (
                    f"{event.get('file_name')}, row {event.get('row_number')}: "
                    f"time={event.get('time')}; "
                    f"x={position.get('x')}; y={position.get('y')}; z={position.get('z')}; "
                    f"floor={event.get('floor')}"
                )
            ],
        }
    rows = event.get("rows", [])
    return {
        "file_name": event.get("file_name"),
        "row_numbers": event.get("row_numbers", []),
        "time": event.get("time"),
        "in_game_time": event.get("in_game_time"),
        "verb": event.get("verb"),
        "object": event.get("object"),
        "shop": event.get("shop"),
        "item": event.get("item"),
        "raw_values": [row_citation(event.get("file_name", ""), row) for row in rows[:3]],
    }


def row_citation(file_name: str, row: dict[str, Any]) -> str:
    row_number = row.get("row_number", "?")
    values = []
    for key in ["time", "actor", "verb", "object", "property", "value", "inGameTime"]:
        value = row.get(key)
        if value:
            values.append(f"{key}={value}")
    if not values:
        values = [f"{key}={value}" for key, value in row.items() if value][:8]
    return f"{file_name}, row {row_number}: " + "; ".join(values)


def events_inside_target_shop(
    candidate_events: list[dict[str, Any]],
    entered: list[dict[str, Any]],
    exited: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    intervals = []
    exits = sorted(exited, key=lambda event: event["time_sort"])
    for enter in sorted(entered, key=lambda event: event["time_sort"]):
        exit_after = next((event for event in exits if event["time_sort"] >= enter["time_sort"]), None)
        if exit_after:
            intervals.append((enter["time_sort"], exit_after["time_sort"]))
    return [
        event for event in candidate_events
        if any(start <= event["time_sort"] <= end for start, end in intervals)
    ]


def event_matches_task_buying(event: dict[str, Any], target_shop: str, target_item: str) -> bool:
    if event.get("shop") not in [None, "", target_shop] and event.get("shop") != target_shop:
        return False
    text = event.get("text", "")
    if strict_target_item_match(event, target_item) or supporting_item_object_match(event, target_item):
        return any(word in text for word in BUYING_WORDS)
    return event.get("shop") == target_shop and any(word in text for word in BUYING_WORDS)


def is_payment(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return "pay" in text and strict_target_item_match(event, target_item)


def is_bag_event(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return (
        is_verb(event, "Place")
        and ("bag" in text or "iteminbag" in text)
        and (strict_target_item_match(event, target_item) or supporting_item_object_match(event, target_item))
    )


def is_validation(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return "validate" in text and (event.get("intention") == target_item or canonical_intention(text) == target_item)


def is_npc_or_ask(event: dict[str, Any], target_shop: str, target_item: str) -> bool:
    text = event.get("text", "")
    if "ask" in text:
        return strict_target_item_match(event, target_item)
    if "interact" in text or "npc" in text:
        return event.get("shop") == target_shop
    return False


def is_npc_interaction(event: dict[str, Any]) -> bool:
    text = event.get("text", "")
    return "interact" in text or "npc" in text


def is_ask_strict_target_item(event: dict[str, Any], target_item: str) -> bool:
    return "ask" in event.get("text", "") and strict_target_item_match(event, target_item)


def is_ask_related_or_wrong_item(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return "ask" in text and related_or_wrong_item_match(event, target_item)


def is_pay_related_or_wrong_item(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return "pay" in text and related_or_wrong_item_match(event, target_item)


def strict_target_item_match(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return target_item in STRICT_TARGET_ITEM_ALIASES and text_matches_aliases(text, STRICT_TARGET_ITEM_ALIASES[target_item])


def supporting_item_object_match(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return target_item in SUPPORTING_ITEM_OBJECT_ALIASES and text_matches_aliases(text, SUPPORTING_ITEM_OBJECT_ALIASES[target_item])


def related_or_wrong_item_match(event: dict[str, Any], target_item: str) -> bool:
    text = event.get("text", "")
    return target_item in RELATED_OR_WRONG_ITEM_ALIASES and text_matches_aliases(text, RELATED_OR_WRONG_ITEM_ALIASES[target_item])


def text_matches_aliases(text: str, aliases: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(alias) in normalized for alias in aliases)


def is_secondary_event(event: dict[str, Any]) -> bool:
    return event.get("object") == "Tag" and event.get("verb") == "Scan"


def unique_event_time_count(events: list[dict[str, Any]]) -> int:
    times = {
        str(event.get("time", "")).strip()
        for event in events
        if str(event.get("time", "")).strip()
    }
    return len(times) if times else len(events)


def is_verb(event: dict[str, Any], verb: str) -> bool:
    return event.get("verb", "").lower() == verb.lower()


def first_property(event: dict[str, Any], name: str) -> str | None:
    values = event.get("properties", {}).get(name) or []
    return values[0] if values else None


def canonical_shop(text: str) -> str | None:
    normalized = normalize_text(text)
    for canonical, aliases in SHOP_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return None


def canonical_item(text: str) -> str | None:
    normalized = normalize_text(text)
    for canonical, aliases in ITEM_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return None


def canonical_intention(text: str) -> str | None:
    normalized = normalize_text(text)
    for canonical, aliases in INTENTION_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return canonical
    return None


def trajectory_note(trajectory_summary: dict[str, Any], trajectory_evidence: dict[str, Any]) -> str:
    if not trajectory_summary.get("available"):
        return "No trajectory CSV was loaded for this participant/scenario."
    if trajectory_evidence.get("status") == "near_inferred_target_shop_area":
        return (
            "Trajectory CSV was loaded. Python inferred the participant came near the target shop area using "
            "a cross-participant shop-position model, but event-log Verb == Enter/Exit remains the source for shop entry/exit."
        )
    return (
        "Trajectory CSV was loaded and compared to the inferred target shop position. "
        "Event-log Verb == Enter/Exit remains the source for shop entry/exit."
    )


def shop_label_from_row(row: dict[str, str], raw_text: str) -> str | None:
    if row.get("object") == "Shop" and row.get("property") == "ShopName" and row.get("value"):
        return normalize_shop_variant(row.get("value", ""))
    normalized = normalize_text(raw_text)
    for aliases in SHOP_ALIASES.values():
        for alias in sorted(aliases, key=len, reverse=True):
            if alias in normalized:
                return normalize_shop_variant(alias)
    return None


def normalize_shop_variant(value: str) -> str:
    text = value.strip()
    lowered = text.lower()
    if lowered == "zara1":
        return "Zara1"
    if lowered == "zara2":
        return "Zara2"
    if lowered == "zara":
        return "Zara"
    if lowered == "fnac":
        return "Fnac"
    if lowered == "decathlon":
        return "Decathlon"
    return text


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return text.lower()


def safe_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def safe_float(value: str | int | float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None
