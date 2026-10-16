from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


TASK_LABELS = [
    "Buy sandals at Decathlon",
    "Buy a travel guide at Fnac",
    "Buy a cap at Zara",
]


@dataclass
class TaskSummary:
    participant_id: str
    task: str
    source: str
    outcome: str = "unknown"
    raw_status: str = "unknown"
    supported_clues: set[int] = field(default_factory=set)
    rejected_clues: set[int] = field(default_factory=set)
    unclear_clues: set[int] = field(default_factory=set)
    clue_details: dict[int, dict[str, str]] = field(default_factory=dict)
    completion_evidence: list[str] = field(default_factory=list)
    final_reason: str = ""
    trajectory_reasoning: str = ""
    warnings: list[str] = field(default_factory=list)


def build_comparison_report(
    *,
    azure_report_path: Path,
    python_report_path: Path,
    foundry_report_path: Path | None = None,
) -> str:
    azure_text = read_if_exists(azure_report_path)
    python_text = read_if_exists(python_report_path)
    foundry_text = read_if_exists(foundry_report_path) if foundry_report_path else ""
    if not azure_text and not python_text and not foundry_text:
        return "# Python vs Azure vs Foundry Comparison Report\n\nNo Python, Azure, or Foundry report files were found.\n"

    azure_tasks = parse_azure_report(azure_text)
    python_tasks = parse_python_report(python_text)
    foundry_tasks = parse_foundry_report(foundry_text)
    return comparison_markdown(
        azure_tasks=azure_tasks,
        python_tasks=python_tasks,
        foundry_tasks=foundry_tasks,
        azure_report_path=azure_report_path,
        python_report_path=python_report_path,
        foundry_report_path=foundry_report_path,
    )


def build_existing_outputs_comparison_report(outputs_dir: Path) -> str:
    pairs = []
    for python_path in sorted(outputs_dir.glob("*_python_cues.md")):
        prefix = python_path.name.removesuffix("_python_cues.md")
        azure_path = outputs_dir / f"{prefix}_azure_report.md"
        foundry_path = outputs_dir / f"{prefix}_foundry_agent_generated_cues.md"
        if azure_path.exists():
            pairs.append((azure_path, python_path, foundry_path if foundry_path.exists() else None))

    lines = ["# Python vs Azure vs Foundry Comparison Report", ""]
    if not pairs:
        lines.extend([
            "No matching `*_azure_report.md` and `*_python_cues.md` pairs were found.",
            "",
        ])
        return "\n".join(lines)

    for azure_path, python_path, foundry_path in pairs:
        report = build_comparison_report(
            azure_report_path=azure_path,
            python_report_path=python_path,
            foundry_report_path=foundry_path,
        )
        lines.extend(report.splitlines()[2:])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def read_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def parse_azure_report(text: str) -> dict[tuple[str, str], TaskSummary]:
    summaries: dict[tuple[str, str], TaskSummary] = {}
    for participant_id, participant_text in split_participant_sections(text).items():
        for task in TASK_LABELS:
            section = find_task_section(participant_text, task, azure=True)
            if not section:
                continue
            summary = TaskSummary(
                participant_id=participant_id,
                task=task,
                source="Azure",
                outcome=normalize_outcome(extract_field(section, "Outcome")),
                raw_status=extract_field(section, "Outcome") or "unknown",
                supported_clues=extract_clues_from_named_section(section, "Supported contextual cues"),
                rejected_clues=extract_clues_from_named_section(section, "Rejected failure cues"),
                unclear_clues=extract_clues_from_named_section(section, "Unclear cues"),
                clue_details=extract_all_clue_details(section),
                completion_evidence=extract_bullets_from_named_section(section, "Completion evidence"),
                final_reason=extract_paragraph_field(section, "Final reason"),
                trajectory_reasoning=extract_paragraph_field(section, "Trajectory reasoning"),
            )
            summary.warnings.extend(find_report_warnings(section))
            summaries[(participant_id, task)] = summary
    return summaries


def parse_python_report(text: str) -> dict[tuple[str, str], TaskSummary]:
    summaries: dict[tuple[str, str], TaskSummary] = {}
    for participant_id, participant_text in split_participant_sections(text).items():
        for task in TASK_LABELS:
            section = find_task_section(participant_text, task, azure=False)
            if not section:
                continue
            raw_status = extract_python_status(section)
            summary = TaskSummary(
                participant_id=participant_id,
                task=task,
                source="Python",
                outcome=normalize_python_status(raw_status),
                raw_status=raw_status or "unknown",
                supported_clues=extract_clues_from_named_section(section, "Supported contextual cues"),
                rejected_clues=extract_clues_from_named_section(section, "Rejected failure cues"),
                unclear_clues=extract_clues_from_named_section(section, "Unclear cues"),
                clue_details=extract_all_clue_details(section),
                completion_evidence=extract_bullets_from_named_section(section, "Completion evidence"),
                final_reason=python_status_reason(raw_status),
                trajectory_reasoning=extract_trajectory_lines(section),
            )
            summary.warnings.extend(find_report_warnings(section))
            summaries[(participant_id, task)] = summary
    return summaries


def parse_foundry_report(text: str) -> dict[tuple[str, str], TaskSummary]:
    summaries: dict[tuple[str, str], TaskSummary] = {}
    for participant_id, participant_text in split_participant_sections(text).items():
        for task in TASK_LABELS:
            section = find_foundry_task_section(participant_text, participant_id, task)
            if not section:
                continue
            raw_outcome = extract_field(section, "- Outcome") or extract_field(section, "Outcome")
            summary = TaskSummary(
                participant_id=participant_id,
                task=task,
                source="Foundry",
                outcome=normalize_outcome(raw_outcome),
                raw_status=raw_outcome or "unknown",
                supported_clues=extract_clues_from_named_section(section, "- Supported cues")
                or extract_clues_from_named_section(section, "Supported cues"),
                rejected_clues=extract_clues_from_named_section(section, "- Rejected or unclear cues")
                or extract_clues_from_named_section(section, "Rejected or unclear cues"),
                unclear_clues=extract_clues_from_named_section(section, "- Rejected or unclear cues")
                or extract_clues_from_named_section(section, "Rejected or unclear cues"),
                clue_details=extract_all_clue_details(section),
                completion_evidence=extract_bullets_from_named_section(section, "- Completion evidence")
                or extract_bullets_from_named_section(section, "Completion evidence"),
                final_reason=extract_paragraph_field(section, "- Final reason")
                or extract_paragraph_field(section, "Final reason"),
                trajectory_reasoning=extract_trajectory_lines(section),
            )
            summary.warnings.extend(find_report_warnings(section))
            summaries[(participant_id, task)] = summary
    return summaries


def split_participant_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+([A-Za-z]{2}\d{2}|am\d{2})\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1)] = text[start:end]
    if not sections:
        participant = re.search(r"Participant:\s*([A-Za-z]{2}\d{2}|am\d{2})", text)
        if participant:
            sections[participant.group(1)] = text
    if not sections:
        participant = re.search(r"^###\s+Participant\s+([A-Za-z]{2}\d{2}|am\d{2})\s+-", text, flags=re.MULTILINE)
        if participant:
            sections[participant.group(1)] = text
    return sections


def find_task_section(text: str, task: str, *, azure: bool) -> str:
    if azure:
        pattern = rf"^Task:\s*{re.escape(task)}\s*$"
    else:
        pattern = rf"^###\s+{re.escape(task)}\s*$"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return ""
    start = matches[0].start()
    next_pattern = r"^Task:\s+|^###\s+"
    next_match = re.search(next_pattern, text[matches[0].end():], flags=re.MULTILINE)
    end = matches[0].end() + next_match.start() if next_match else len(text)
    return text[start:end]


def find_foundry_task_section(text: str, participant_id: str, task: str) -> str:
    variants = {task}
    if task == "Buy a travel guide at Fnac":
        variants.add("Buy travel guide at Fnac")
    for variant in variants:
        pattern = rf"^###\s+Participant\s+{re.escape(participant_id)}\s+-\s+{re.escape(variant)}\s*$"
        matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
        if not matches:
            continue
        start = matches[0].start()
        next_match = re.search(r"^###\s+Participant\s+", text[matches[0].end():], flags=re.MULTILINE)
        end = matches[0].end() + next_match.start() if next_match else len(text)
        return text[start:end]
    return ""


def extract_field(section: str, field_name: str) -> str:
    escaped = re.escape(field_name)
    match = re.search(rf"^{escaped}:\s*(.+?)\s*$", section, flags=re.MULTILINE)
    if not match and field_name.startswith("- "):
        match = re.search(rf"^{escaped}:\s*(.+?)\s*$", section, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def extract_python_status(section: str) -> str:
    match = re.search(r"Python task status:\s*`([^`]+)`", section)
    return match.group(1).strip() if match else ""


def normalize_outcome(value: str) -> str:
    lower = value.lower()
    if "completed" in lower:
        return "completed"
    if "wrong" in lower:
        return "wrong item"
    if "failed" in lower or "not completed" in lower:
        return "failed"
    if "unclear" in lower:
        return "unclear"
    return "unknown"


def normalize_python_status(value: str) -> str:
    lower = value.lower()
    if lower == "completed_evidence_found":
        return "completed"
    if "wrong" in lower:
        return "wrong item"
    if lower and lower != "unknown":
        return "failed"
    return "unknown"


def extract_clues_from_named_section(section: str, heading: str) -> set[int]:
    block = extract_named_section(section, heading)
    return {int(match.group(1)) for match in re.finditer(r"Clue\s+(\d+)", block)}


def extract_all_clue_details(section: str) -> dict[int, dict[str, str]]:
    details: dict[int, dict[str, str]] = {}
    for heading, status in [
        ("Supported contextual cues", "supported"),
        ("Rejected failure cues", "rejected"),
        ("Unclear cues", "unclear"),
    ]:
        block = extract_named_section(section, heading)
        for number, clue_block in extract_clue_blocks(block).items():
            details[number] = {
                "status": status,
                "reason": extract_reason_from_clue_block(clue_block),
                "evidence": summarize_evidence_from_clue_block(clue_block),
            }
    return details


def extract_clue_blocks(block: str) -> dict[int, str]:
    matches = list(re.finditer(r"(?m)^-?\s*Clue\s+(\d+)\b.*$", block))
    result: dict[int, str] = {}
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        result[int(match.group(1))] = block[start:end].strip()
    return result


def extract_reason_from_clue_block(block: str) -> str:
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if re.search(r"\bReason:\s*$", line):
            reason_lines = []
            for next_line in lines[index + 1:]:
                stripped = next_line.strip()
                if not stripped:
                    if reason_lines:
                        break
                    continue
                if stripped.startswith("- "):
                    reason_lines.append(stripped[2:].strip())
                else:
                    reason_lines.append(stripped)
                if len(reason_lines) >= 2:
                    break
            if reason_lines:
                return " ".join(reason_lines)
        inline = re.search(r"Reason:\s*(.+)", line)
        if inline:
            return inline.group(1).strip()
    return first_meaningful_line(block)


def summarize_evidence_from_clue_block(block: str) -> str:
    evidence_lines = []
    in_evidence = False
    for line in block.splitlines():
        stripped = line.strip()
        if re.search(r"\bEvidence:\s*$", stripped):
            in_evidence = True
            continue
        if in_evidence and re.search(r"\bReason:\s*$", stripped):
            break
        if in_evidence and stripped.startswith("- "):
            evidence_lines.append(stripped[2:].strip())
        if len(evidence_lines) >= 2:
            break
    return " ".join(evidence_lines)


def extract_bullets_from_named_section(section: str, heading: str) -> list[str]:
    block = extract_named_section(section, heading)
    bullets = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
        if len(bullets) >= 5:
            break
    return bullets


def extract_paragraph_field(section: str, heading: str) -> str:
    block = extract_named_section(section, heading)
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("---"):
            lines.append(stripped)
        if len(lines) >= 4:
            break
    return " ".join(lines)


def extract_trajectory_lines(section: str) -> str:
    lines = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("- Trajectory evidence:") or stripped.startswith("- Trajectory note:"):
            lines.append(stripped[2:].strip())
        if len(lines) >= 2:
            break
    return " ".join(lines)


def python_status_reason(raw_status: str) -> str:
    if raw_status == "completed_evidence_found":
        return "Python found completion evidence for the task."
    if raw_status == "watched_or_stopped_at_shop_but_no_enter_evidence":
        return "Python found Watch/StartToWatch/StopToWatch evidence, but no real Verb == Enter evidence for the target shop."
    if raw_status:
        return f"Python classified the task as `{raw_status}`."
    return ""


def first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" -")
        if stripped:
            return stripped
    return ""


def extract_named_section(section: str, heading: str) -> str:
    match = re.search(rf"^{re.escape(heading)}:\s*$", section, flags=re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(
        r"^(-\s+)?(Completion evidence|Supported contextual cues|Supported cues|Rejected failure cues|Rejected or unclear cues|Unclear cues|Final reason|Confidence|Caution|Trajectory reasoning):\s*$",
        section[start:],
        flags=re.MULTILINE,
    )
    end = start + next_heading.start() if next_heading else len(section)
    return section[start:end]


def find_report_warnings(section: str) -> list[str]:
    warnings = []
    supported = extract_named_section(section, "Supported contextual cues").lower()
    if any(word in supported for word in ["rejected", "not supported", "not applicable"]):
        warnings.append("Supported contextual cues section contains rejected/not-supported wording.")
    if "strict time limit" in section.lower():
        warnings.append("Report may still be using an overly strict immediacy rule.")
    return warnings


def comparison_markdown(
    *,
    azure_tasks: dict[tuple[str, str], TaskSummary],
    python_tasks: dict[tuple[str, str], TaskSummary],
    foundry_tasks: dict[tuple[str, str], TaskSummary],
    azure_report_path: Path,
    python_report_path: Path,
    foundry_report_path: Path | None,
) -> str:
    include_foundry = foundry_report_path is not None
    keys = sorted(set(azure_tasks) | set(python_tasks) | set(foundry_tasks))
    lines = [
        "# Python vs Azure Comparison Report" if not include_foundry else "# Python vs Azure vs Foundry Comparison Report",
        "",
        f"Azure report: `{azure_report_path}`",
        f"Python report: `{python_report_path}`",
        "",
    ]
    if include_foundry:
        lines.insert(4, f"Foundry report: `{foundry_report_path}`")
    if not keys:
        lines.extend(["No comparable participant/task sections were found.", ""])
        return "\n".join(lines)

    for participant_id in sorted({key[0] for key in keys}):
        lines.extend([f"## {participant_id}", ""])
        for task in TASK_LABELS:
            key = (participant_id, task)
            azure = azure_tasks.get(key)
            python = python_tasks.get(key)
            foundry = foundry_tasks.get(key)
            if not azure and not python and not foundry:
                continue
            lines.extend(compare_task_markdown(task, azure, python, foundry, include_foundry=include_foundry))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compare_task_markdown(
    task: str,
    azure: TaskSummary | None,
    python: TaskSummary | None,
    foundry: TaskSummary | None = None,
    *,
    include_foundry: bool = False,
) -> list[str]:
    lines = [f"### {task}", ""]
    if not azure:
        lines.extend([
            "**Outcome Comparison**",
            "",
            "- Azure outcome: `missing`",
            f"- Python outcome: `{python.outcome}` (`{python.raw_status}`)",
            "- Outcome agreement: `not comparable`",
            "",
            "**Why Azure Is Missing**",
            "",
            "- The Azure report did not contain a task-level section for this task. This usually means the Azure API call failed or returned an incomplete/error report.",
            "",
            "**Python Reasoning Snapshot**",
            "",
            f"- Python status meaning: {python.final_reason or 'not found in report'}",
            f"- Completion evidence examples: {format_list(python.completion_evidence)}",
            f"- Trajectory reasoning: {python.trajectory_reasoning or 'not found in report'}",
            "",
            "**Python Supported Clues**",
            "",
            f"- Supported by Python: `{format_clues(python.supported_clues)}`",
            "",
        ])
        if python.clue_details:
            lines.extend(["**Python Per-Clue Details**", ""])
            for clue_number in sorted(python.clue_details):
                detail = python.clue_details[clue_number]
                lines.extend([
                    f"- Clue {clue_number}: `{detail.get('status', 'not mentioned')}`",
                    f"  - Reason: {detail.get('reason', 'no reason found')}",
                ])
                if detail.get("evidence"):
                    lines.append(f"  - Evidence: {detail['evidence']}")
        lines.extend([
            "",
            "**Comparison Note**",
            "",
            "- Azure did not produce analyzable task details, so only the Python clue analysis can be reviewed for this task.",
        ])
        return lines
    if not python:
        lines.extend([
            "**Outcome Comparison**",
            "",
            f"- Azure outcome: `{azure.outcome}` (`{azure.raw_status}`)",
            "- Python outcome: `missing`",
            "- Outcome agreement: `not comparable`",
            "",
            "**Azure Reasoning Snapshot**",
            "",
            f"- Final reason: {azure.final_reason or 'not found in report'}",
            f"- Completion evidence examples: {format_list(azure.completion_evidence)}",
            f"- Trajectory reasoning: {azure.trajectory_reasoning or 'not found in report'}",
            "",
            "**Comparison Note**",
            "",
            "- Python did not produce analyzable task details, so only the Azure analysis can be reviewed for this task.",
        ])
        return lines

    outcome_agreement = azure.outcome == python.outcome
    foundry_agreement = bool(foundry and foundry.outcome == python.outcome)
    supported_overlap = azure.supported_clues & python.supported_clues
    only_azure = azure.supported_clues - python.supported_clues
    only_python = python.supported_clues - azure.supported_clues

    lines.extend([
        "**Outcome Comparison**",
        "",
        f"- Azure outcome: `{azure.outcome}` (`{azure.raw_status}`)",
        f"- Python outcome: `{python.outcome}` (`{python.raw_status}`)",
        f"- Outcome agreement: `{agreement_label(outcome_agreement)}`",
    ])
    if include_foundry:
        lines.extend([
            f"- Foundry outcome: `{foundry.outcome if foundry else 'missing'}` (`{foundry.raw_status if foundry else 'missing'}`)",
            f"- Foundry agrees with Python: `{agreement_label(foundry_agreement) if foundry else 'not comparable'}`",
        ])
    lines.extend([
        "",
        "**Azure Reasoning Snapshot**",
        "",
        f"- Final reason: {azure.final_reason or 'not found in report'}",
        f"- Completion evidence examples: {format_list(azure.completion_evidence)}",
        f"- Trajectory reasoning: {azure.trajectory_reasoning or 'not found in report'}",
        "",
        "**Python Reasoning Snapshot**",
        "",
        f"- Python status meaning: {python.final_reason or 'not found in report'}",
        f"- Completion evidence examples: {format_list(python.completion_evidence)}",
        f"- Trajectory reasoning: {python.trajectory_reasoning or 'not found in report'}",
        "",
        "**Supported Clue Comparison**",
        "",
        f"- Supported clues in both: `{format_clues(supported_overlap)}`",
        f"- Supported only by Azure: `{format_clues(only_azure)}`",
        f"- Supported only by Python: `{format_clues(only_python)}`",
    ])
    if include_foundry:
        lines[lines.index("**Supported Clue Comparison**") - 1:lines.index("**Supported Clue Comparison**") - 1] = [
            "",
            "**Foundry Reasoning Snapshot**",
            "",
            f"- Final reason: {foundry.final_reason if foundry else 'not found in report'}",
            f"- Completion evidence examples: {format_list(foundry.completion_evidence) if foundry else 'none found'}",
            f"- Trajectory reasoning: {foundry.trajectory_reasoning if foundry else 'not found in report'}",
        ]

    clue_numbers = sorted(set(azure.clue_details) | set(python.clue_details))
    if clue_numbers:
        lines.extend(["", "**Per-Clue Details**", ""])
        for clue_number in clue_numbers:
            azure_detail = azure.clue_details.get(clue_number, {})
            python_detail = python.clue_details.get(clue_number, {})
            if same_detail_status(azure_detail, python_detail) and clue_number not in supported_overlap:
                continue
            lines.extend([
                f"- Clue {clue_number}",
                f"  - Azure: `{azure_detail.get('status', 'not mentioned')}`; {azure_detail.get('reason', 'no reason found')}",
                f"  - Python: `{python_detail.get('status', 'not mentioned')}`; {python_detail.get('reason', 'no reason found')}",
            ])
            if azure_detail.get("evidence") or python_detail.get("evidence"):
                lines.append(f"  - Evidence contrast: Azure: {azure_detail.get('evidence', 'none')} Python: {python_detail.get('evidence', 'none')}")

    notes = []
    if not outcome_agreement:
        notes.append("Outcome mismatch: inspect raw evidence before presenting this participant.")
    if foundry and not foundry_agreement:
        notes.append("Foundry and Python disagree: inspect the raw rows and any truncated Foundry context.")
    if azure.outcome == "completed" and python.outcome != "completed":
        notes.append("Azure claims completion while Python found no explicit completion evidence. Treat Azure completion as suspect unless the cited raw rows show Ask/Pay/ItemInBag/Validate for the target item.")
    if only_azure or only_python:
        notes.append("Supported clue mismatch: the two methods explain the behavior differently.")
    warnings = azure.warnings + python.warnings
    notes.extend(warnings)
    if not notes:
        notes.append("Azure and Python broadly agree for this task.")
    lines.extend(["", "**Comparison Note**", "", "- " + " ".join(dedupe(notes))])
    return lines


def agreement_label(value: bool) -> str:
    return "agree" if value else "disagree"


def format_clues(clues: set[int]) -> str:
    if not clues:
        return "none"
    return ", ".join(f"Clue {number}" for number in sorted(clues))


def format_list(items: list[str]) -> str:
    if not items:
        return "none found"
    return " | ".join(items[:4])


def same_detail_status(left: dict[str, str], right: dict[str, str]) -> bool:
    return left.get("status") == right.get("status")


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
