from __future__ import annotations

import argparse
import json
from pathlib import Path

from .albert_reasoner import albert_report_markdown, run_albert_raw_reasoner
from .azure_openai_reasoner import (
    audit_azure_evidence_counts,
    azure_openai_report_markdown,
    build_verified_evidence_by_participant,
    run_azure_openai_raw_reasoner,
)
from .cue_engine import calculate_python_cues, python_cues_markdown
from .cue_reasoner import (
    combined_python_evidence_reasoning_markdown,
    python_cue_report_markdown,
    run_albert_python_cue_reasoner,
    run_azure_openai_python_cue_reasoner,
)
from .env_loader import load_dotenv
from .foundry_agent_cue_discovery import (
    build_foundry_participant_context,
    build_foundry_raw_data_context,
    foundry_agent_report_markdown,
    run_foundry_agent_cue_discovery,
)
from .gemini_reasoner import gemini_report_markdown, run_gemini_raw_reasoner
from .local_reasoner import local_report_markdown, run_local_raw_reasoner
from .prompt_loader import load_prompt
from .raw_loader import load_json, load_raw_batch, normalize_environment
from .report_comparer import build_comparison_report, build_existing_outputs_comparison_report


def extract_foundry_analyst_cue_block(text: str) -> str:
    marker = "## ANALYST APPROVAL BLOCK - EDIT THIS SECTION"
    start = text.find(marker)
    if start < 0:
        return text
    block = text[start:]
    for end_marker in ["\n## Analyst Edit Request", "\n## Analyst Validation Request"]:
        end = block.find(end_marker)
        if end >= 0:
            return block[:end].strip()
    return block.strip()


def audit_single_participant_evidence_counts(
    result: dict,
    *,
    raw_batch: dict,
    tasks_config: dict,
    clues_config: dict,
) -> dict:
    """Reject a one-participant LLM report when its numeric evidence table contradicts raw CSV counts."""
    participants = raw_batch.get("participants", [])
    if len(participants) != 1:
        return result
    participant_id = str(participants[0].get("participant_id", ""))
    verified = build_verified_evidence_by_participant(
        raw_batch=raw_batch,
        tasks_config=tasks_config,
        clues_config=clues_config,
    )
    return audit_azure_evidence_counts(result, verified.get(participant_id, {}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run raw-LLM or Python-clue prospective-memory VR analysis experiments."
    )
    parser.add_argument("--participants-root", type=Path, default=Path("participants"))
    parser.add_argument("--tasks", type=Path, default=Path("config/tasks.json"))
    parser.add_argument("--clues", type=Path, default=Path("config/clues.json"))
    parser.add_argument("--shop-locations", type=Path, default=Path("config/shop_locations.json"))
    parser.add_argument("--raw-reasoner-prompt", type=Path, default=Path("config/raw_reasoner_prompt.md"))
    parser.add_argument("--python-cue-reasoner-prompt", type=Path, default=Path("config/python_cue_reasoner_prompt.md"))
    parser.add_argument("--cue-discovery-domain", type=Path, default=Path("config/cue_discovery_domain.md"))
    parser.add_argument("--cue-discovery-prompt", type=Path, default=Path("config/cue_discovery_prompt.md"))
    parser.add_argument("--out", type=Path, default=Path("outputs/raw_payload.json"))
    parser.add_argument("--albert-out", type=Path, default=Path("outputs/albert_raw_report.md"))
    parser.add_argument("--azure-openai-out", type=Path, default=Path("outputs/azure_openai_raw_report.md"))
    parser.add_argument("--gemini-out", type=Path, default=Path("outputs/gemini_raw_report.md"))
    parser.add_argument("--local-out", type=Path, default=Path("outputs/local_gpt_oss_report.md"))
    parser.add_argument("--python-cues-out", type=Path, default=Path("outputs/python_cues.json"))
    parser.add_argument("--python-cues-md-out", type=Path, default=Path("outputs/python_cues.md"))
    parser.add_argument("--no-python-cues-json", action="store_true", help="Do not write the Python clue JSON file; write only the Markdown clue report.")
    parser.add_argument("--albert-python-cues-out", type=Path, default=Path("outputs/albert_python_cue_report.md"))
    parser.add_argument("--azure-openai-python-cues-out", type=Path, default=Path("outputs/azure_openai_python_cue_report.md"))
    parser.add_argument("--comparison-out", type=Path, default=Path("outputs/azure_python_comparison_report.md"))
    parser.add_argument("--foundry-agent-cue-discovery-out", type=Path, default=Path("outputs/foundry_agent_generated_cues.md"))
    parser.add_argument("--foundry-comparison-report", type=Path, help="Optional Foundry report to include in Azure/Python comparison.")
    parser.add_argument("--compare-existing-outputs", action="store_true", help="Compare existing `*_azure_report.md` and `*_python_cues.md` files in the output directory, then exit.")
    parser.add_argument("--use-albert", action="store_true", help="Send raw participant data to Albert for clue-based reasoning.")
    parser.add_argument("--use-azure-openai", action="store_true", help="Send raw participant data to Azure OpenAI for clue-based reasoning.")
    parser.add_argument("--use-foundry-agent-cue-discovery", action="store_true", help="Use Azure AI Foundry Agent Service for analyst-guided clue expansion, AI clue generation, validation, and reasoning.")
    parser.add_argument("--use-gemini", action="store_true", help="Send raw participant data to Gemini for clue-based reasoning.")
    parser.add_argument("--use-local", action="store_true", help="Send raw participant data to a local OpenAI-compatible model server such as Ollama.")
    parser.add_argument("--calculate-python-cues", action="store_true", help="Calculate clue evidence in Python and write it as JSON.")
    parser.add_argument("--use-albert-python-cues", action="store_true", help="Send Python-calculated clue evidence to Albert for reasoning.")
    parser.add_argument("--use-azure-openai-python-cues", action="store_true", help="Send Python-calculated clue evidence to Azure OpenAI for reasoning.")
    parser.add_argument("--albert-model", default="openai/gpt-oss-120b", help="Albert model name.")
    parser.add_argument("--albert-sleep-seconds", type=float, default=7.0, help="Delay between Albert calls to reduce rate-limit errors.")
    parser.add_argument("--albert-chunk-rows", type=int, default=0, help="If set, split each raw CSV file into chunks of this many data rows before calling Albert.")
    parser.add_argument("--albert-chunk-overlap-rows", type=int, default=25, help="Repeat this many rows from the previous chunk for context.")
    parser.add_argument("--albert-max-chunks", type=int, default=0, help="Maximum chunks per participant for testing. Default 0 means all chunks.")
    parser.add_argument("--albert-retry-count", type=int, default=3, help="Retry Albert chunk calls this many times after quota/rate errors.")
    parser.add_argument("--albert-retry-sleep-seconds", type=float, default=90.0, help="Wait this many seconds before retrying a quota/rate-limited Albert chunk call.")
    parser.add_argument("--albert-python-cues-timeout-seconds", type=float, default=600.0, help="Timeout for the Albert call over Python-calculated clues.")
    parser.add_argument("--azure-openai-deployment", help="Azure OpenAI deployment name. Defaults to AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--azure-openai-api-version", help="Azure OpenAI API version. Defaults to AZURE_OPENAI_API_VERSION or 2024-10-21.")
    parser.add_argument("--azure-openai-sleep-seconds", type=float, default=0.0, help="Delay between Azure OpenAI calls when analyzing multiple participants.")
    parser.add_argument("--azure-openai-timeout-seconds", type=float, default=900.0, help="Timeout for each Azure OpenAI call.")
    parser.add_argument("--azure-openai-chunk-rows", type=int, default=0, help="Force Azure OpenAI chunking with this many data rows per chunk. Default 0 tries one full raw request first and automatically falls back to 1000-row chunks only if Azure returns a context-length error.")
    parser.add_argument("--azure-openai-chunk-overlap-rows", type=int, default=25, help="Repeat this many rows from the previous Azure OpenAI chunk for context.")
    parser.add_argument("--azure-openai-max-chunks", type=int, default=0, help="Maximum Azure OpenAI chunks per participant for testing. Default 0 means all chunks.")
    parser.add_argument("--cue-discovery-context-chars", type=int, default=0, help="Maximum characters of existing participant reports to include as context for clue discovery. Default 0 keeps clue discovery independent from old reports.")
    parser.add_argument("--foundry-raw-context-chars", type=int, default=180000, help="Maximum raw CSV characters to send to Foundry clue discovery. Foundry has a hard per-message size limit.")
    parser.add_argument("--foundry-project-endpoint", help="Azure AI Foundry project endpoint. Defaults to PROJECT_ENDPOINT or AZURE_AI_PROJECT_ENDPOINT.")
    parser.add_argument("--foundry-model-deployment", help="Azure AI Foundry model deployment name. Defaults to MODEL_DEPLOYMENT_NAME or AZURE_OPENAI_DEPLOYMENT.")
    parser.add_argument("--foundry-auth-mode", choices=["auto", "api_key", "azure_identity"], default=None, help="Foundry auth mode. Default auto uses PROJECT_API_KEY if present, otherwise Azure login.")
    parser.add_argument("--foundry-workflow-stage", choices=["cue_validation", "final_reasoner", "full"], default="full", help="Foundry stage: cue_validation prepares clues for analyst validation, final_reasoner uses validated clues, full does both in one report.")
    parser.add_argument("--foundry-validated-cues", type=Path, help="Markdown or text file containing analyst-validated clues for --foundry-workflow-stage final_reasoner.")
    parser.add_argument("--foundry-interactive-validation", action="store_true", help="Run Foundry clue validation first, pause for analyst Y/N approval in the terminal, then run final reasoning using the reviewed clue file.")
    parser.add_argument("--foundry-final-reasoner-out", type=Path, help="Output Markdown path for the Foundry final reasoner when --foundry-interactive-validation is used. Defaults to a sibling file ending in _final_reasoner.md.")
    parser.add_argument("--keep-foundry-agent", action="store_true", help="Keep the temporary Foundry Agent after the run instead of deleting it.")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name.")
    parser.add_argument("--gemini-sleep-seconds", type=float, default=0.0, help="Delay between Gemini calls when analyzing multiple participants.")
    parser.add_argument("--gemini-timeout-seconds", type=float, default=600.0, help="Timeout for each Gemini call.")
    parser.add_argument("--local-model", default="gpt-oss:20b", help="Local model name, for example gpt-oss:20b for Ollama.")
    parser.add_argument("--local-base-url", default="http://localhost:11434/v1", help="OpenAI-compatible local server URL.")
    parser.add_argument("--local-sleep-seconds", type=float, default=0.0, help="Delay between local model calls.")
    parser.add_argument("--local-chunk-rows", type=int, default=300, help="Split each raw CSV file into chunks of this many data rows for the local model.")
    parser.add_argument("--local-chunk-overlap-rows", type=int, default=25, help="Repeat this many rows from the previous local chunk for context.")
    parser.add_argument("--local-max-chunks", type=int, default=0, help="Maximum local chunks per participant for testing. Default 0 means all chunks.")
    parser.add_argument("--local-timeout-seconds", type=float, default=900.0, help="Timeout for each local model call.")
    parser.add_argument("--participant", help="Only load/analyze one participant, for example AG04.")
    parser.add_argument("--scenario", help="Only load one scenario, for example ShoppingCenter, House, or Tutorial.")
    parser.add_argument("--max-participants", type=int, default=0, help="Maximum participants to load/analyze. Default 0 means no limit.")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="Limit raw rows per CSV file for testing. Default 0 means all rows.")
    parser.add_argument("--keep-all-sessions", action="store_true", help="Load every CSV session instead of keeping only the longest session per scenario/data kind.")
    parser.add_argument("--exclude-tutorial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--exclude-trajectory", action="store_true", help="Do not include spatiotemporal trajectory CSV files.")
    parser.add_argument("--no-house-for-python-cues", action="store_true", help="Do not add House logs to Python clue analysis when --scenario ShoppingCenter is used.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    load_dotenv()

    if args.compare_existing_outputs:
        args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_out.write_text(
            build_existing_outputs_comparison_report(args.comparison_out.parent),
            encoding="utf-8"
        )
        print(f"Wrote {args.comparison_out}")
        return

    tasks_config = load_json(args.tasks)
    clues_config = load_json(args.clues)
    shop_locations_config = load_json(args.shop_locations)
    tasks_context = {
        **tasks_config,
        "shop_locations": shop_locations_config,
    }
    raw_reasoner_prompt = load_prompt(args.raw_reasoner_prompt)
    python_cue_reasoner_prompt = load_prompt(args.python_cue_reasoner_prompt)
    cue_discovery_domain = load_prompt(args.cue_discovery_domain)
    cue_discovery_prompt = load_prompt(args.cue_discovery_prompt)
    scenario_filter = normalize_environment(args.scenario) if args.scenario else None
    if args.scenario and scenario_filter is None:
        raise SystemExit(f"Unknown scenario: {args.scenario}")
    if (
        args.use_foundry_agent_cue_discovery
        and not args.foundry_interactive_validation
        and args.foundry_workflow_stage == "final_reasoner"
        and not args.foundry_validated_cues
    ):
        raise SystemExit(
            "--foundry-workflow-stage final_reasoner requires --foundry-validated-cues. "
            "First run --foundry-workflow-stage cue_validation, review the clue proposal, "
            "then save approved clues to a Markdown file such as config/foundry_validated_cues.md."
        )
    if (
        args.foundry_validated_cues
        and not args.foundry_interactive_validation
        and not args.foundry_validated_cues.exists()
    ):
        raise SystemExit(
            f"Validated clues file not found: {args.foundry_validated_cues}. "
            "Create this file after analyst validation. You can start from "
            "config/foundry_validated_cues.example.md."
        )

    raw_batch = load_raw_batch(
        args.participants_root,
        participant_filter=args.participant,
        scenario_filter=scenario_filter,
        max_participants=args.max_participants,
        include_tutorial=False,
        include_trajectory=not args.exclude_trajectory,
        max_rows_per_file=args.max_rows_per_file,
        longest_session_only=not args.keep_all_sessions,
    )

    payload_config = {"tasks": tasks_context}
    foundry_only_run = (
        args.use_foundry_agent_cue_discovery
        and not args.use_albert
        and not args.use_azure_openai
        and not args.use_gemini
        and not args.use_local
        and not args.calculate_python_cues
        and not args.use_albert_python_cues
        and not args.use_azure_openai_python_cues
    )
    if not foundry_only_run:
        payload_config["clues"] = clues_config

    payload = {
        "loader_role": "Python only loaded raw CSV rows and config. It did not group events, test clues, or decide outcomes.",
        "config": payload_config,
        "raw_data": raw_batch
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

    if args.use_foundry_agent_cue_discovery:
        participant_context = ""
        if args.cue_discovery_context_chars > 0:
            participant_context = build_foundry_participant_context(
                participant_id=args.participant,
                outputs_dir=args.foundry_agent_cue_discovery_out.parent,
                max_characters=args.cue_discovery_context_chars,
            )

        if args.foundry_interactive_validation:
            validation_result = run_foundry_agent_cue_discovery(
                domain_text=cue_discovery_domain,
                discovery_prompt=cue_discovery_prompt,
                tasks_config=tasks_context,
                clues_config=clues_config,
                raw_data_context=build_foundry_raw_data_context(
                    raw_batch=raw_batch,
                    max_characters=args.foundry_raw_context_chars,
                    shop_locations_config=shop_locations_config,
                ),
                participant_context=participant_context,
                workflow_stage="cue_validation",
                validated_cues_context="",
                project_endpoint=args.foundry_project_endpoint,
                model_deployment_name=args.foundry_model_deployment,
                auth_mode=args.foundry_auth_mode,
                cleanup_agent=not args.keep_foundry_agent,
            )
            args.foundry_agent_cue_discovery_out.parent.mkdir(parents=True, exist_ok=True)
            args.foundry_agent_cue_discovery_out.write_text(
                foundry_agent_report_markdown(validation_result),
                encoding="utf-8"
            )
            print(f"Wrote {args.foundry_agent_cue_discovery_out}")
            print(
                "\nReview this clue proposal file now. Rename, overwrite, edit, delete, or add clues in the section named "
                "`ANALYST APPROVAL BLOCK - EDIT THIS SECTION`; the final reasoner "
                "will use only the clues left in that block.\n"
                f"{args.foundry_agent_cue_discovery_out}\n"
            )
            approval = input("Use this analyst-edited clue file for Foundry final reasoning? [y/N]: ").strip().lower()
            if approval not in {"y", "yes"}:
                print("Stopped before Foundry final reasoning. Clue proposal file is saved for analyst editing.")
                return

            validated_cues_context = extract_foundry_analyst_cue_block(
                args.foundry_agent_cue_discovery_out.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
            final_reasoner_out = args.foundry_final_reasoner_out or args.foundry_agent_cue_discovery_out.with_name(
                f"{args.foundry_agent_cue_discovery_out.stem}_final_reasoner.md"
            )
            final_result = run_foundry_agent_cue_discovery(
                domain_text=cue_discovery_domain,
                discovery_prompt=cue_discovery_prompt,
                tasks_config=tasks_context,
                clues_config=clues_config,
                raw_data_context=build_foundry_raw_data_context(
                    raw_batch=raw_batch,
                    max_characters=args.foundry_raw_context_chars,
                    shop_locations_config=shop_locations_config,
                ),
                participant_context=participant_context,
                workflow_stage="final_reasoner",
                validated_cues_context=validated_cues_context,
                project_endpoint=args.foundry_project_endpoint,
                model_deployment_name=args.foundry_model_deployment,
                auth_mode=args.foundry_auth_mode,
                cleanup_agent=not args.keep_foundry_agent,
            )
            final_result = audit_single_participant_evidence_counts(
                final_result,
                raw_batch=raw_batch,
                tasks_config=tasks_context,
                clues_config=clues_config,
            )
            final_reasoner_out.parent.mkdir(parents=True, exist_ok=True)
            final_reasoner_out.write_text(
                foundry_agent_report_markdown(final_result),
                encoding="utf-8"
            )
            print(f"Wrote {final_reasoner_out}")
        else:
            validated_cues_context = ""
            if args.foundry_validated_cues:
                validated_cues_context = args.foundry_validated_cues.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            foundry_result = run_foundry_agent_cue_discovery(
                domain_text=cue_discovery_domain,
                discovery_prompt=cue_discovery_prompt,
                tasks_config=tasks_context,
                clues_config=clues_config,
                raw_data_context=build_foundry_raw_data_context(
                    raw_batch=raw_batch,
                    max_characters=args.foundry_raw_context_chars,
                    shop_locations_config=shop_locations_config,
                ),
                participant_context=participant_context,
                workflow_stage=args.foundry_workflow_stage,
                validated_cues_context=validated_cues_context,
                project_endpoint=args.foundry_project_endpoint,
                model_deployment_name=args.foundry_model_deployment,
                auth_mode=args.foundry_auth_mode,
                cleanup_agent=not args.keep_foundry_agent,
            )
            if args.foundry_workflow_stage in {"final_reasoner", "full"}:
                foundry_result = audit_single_participant_evidence_counts(
                    foundry_result,
                    raw_batch=raw_batch,
                    tasks_config=tasks_context,
                    clues_config=clues_config,
                )
            args.foundry_agent_cue_discovery_out.parent.mkdir(parents=True, exist_ok=True)
            args.foundry_agent_cue_discovery_out.write_text(
                foundry_agent_report_markdown(foundry_result),
                encoding="utf-8"
            )
            print(f"Wrote {args.foundry_agent_cue_discovery_out}")

    python_cues = None
    needs_python_cues = args.calculate_python_cues or args.use_albert_python_cues or args.use_azure_openai_python_cues
    if needs_python_cues:
        python_cue_raw_batch = raw_batch
        if scenario_filter == "ShoppingCenter" and not args.no_house_for_python_cues:
            python_cue_raw_batch = load_raw_batch(
                args.participants_root,
                participant_filter=args.participant,
                scenario_filter=None,
                max_participants=args.max_participants,
                include_tutorial=False,
                include_trajectory=not args.exclude_trajectory,
                max_rows_per_file=args.max_rows_per_file,
                longest_session_only=not args.keep_all_sessions,
            )
            for participant in python_cue_raw_batch.get("participants", []):
                participant["environments"] = {
                    name: environment
                    for name, environment in participant.get("environments", {}).items()
                    if name in {"House", "ShoppingCenter"}
                }

        trajectory_reference_batch = raw_batch
        if not args.exclude_trajectory:
            trajectory_reference_batch = load_raw_batch(
                args.participants_root,
                scenario_filter="ShoppingCenter",
                include_tutorial=False,
                include_trajectory=True,
                max_rows_per_file=0,
                longest_session_only=not args.keep_all_sessions,
            )
        python_cues = calculate_python_cues(
            raw_batch=python_cue_raw_batch,
            tasks_config=tasks_context,
            clues_config=clues_config,
            reference_batch=trajectory_reference_batch,
            shop_locations_config=shop_locations_config
        )
        should_write_python_cues = args.calculate_python_cues or args.use_albert_python_cues or args.use_azure_openai_python_cues
        if should_write_python_cues and not args.no_python_cues_json:
            args.python_cues_out.parent.mkdir(parents=True, exist_ok=True)
            args.python_cues_out.write_text(json.dumps(python_cues, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {args.python_cues_out}")
        write_standalone_python_cues_md = (
            should_write_python_cues
            and (
                args.calculate_python_cues
                or (not args.use_albert_python_cues and not args.use_azure_openai_python_cues)
            )
        )
        if write_standalone_python_cues_md:
            args.python_cues_md_out.parent.mkdir(parents=True, exist_ok=True)
            args.python_cues_md_out.write_text(python_cues_markdown(python_cues), encoding="utf-8")
            print(f"Wrote {args.python_cues_md_out}")

        if args.use_albert_python_cues:
            albert_python_cue_result = run_albert_python_cue_reasoner(
                cue_batch=python_cues,
                model=args.albert_model,
                system_prompt=python_cue_reasoner_prompt,
                timeout_seconds=args.albert_python_cues_timeout_seconds
            )
            args.albert_python_cues_out.parent.mkdir(parents=True, exist_ok=True)
            args.albert_python_cues_out.write_text(
                combined_python_evidence_reasoning_markdown(
                    cue_evidence_markdown=python_cues_markdown(python_cues),
                    reasoner_result=albert_python_cue_result,
                ),
                encoding="utf-8"
            )
            print(f"Wrote {args.albert_python_cues_out}")

        if args.use_azure_openai_python_cues:
            azure_python_cue_result = run_azure_openai_python_cue_reasoner(
                cue_batch=python_cues,
                deployment=args.azure_openai_deployment,
                api_version=args.azure_openai_api_version,
                system_prompt=python_cue_reasoner_prompt,
                timeout_seconds=args.azure_openai_timeout_seconds,
            )
            args.azure_openai_python_cues_out.parent.mkdir(parents=True, exist_ok=True)
            args.azure_openai_python_cues_out.write_text(
                combined_python_evidence_reasoning_markdown(
                    cue_evidence_markdown=python_cues_markdown(python_cues),
                    reasoner_result=azure_python_cue_result,
                ),
                encoding="utf-8"
            )
            print(f"Wrote {args.azure_openai_python_cues_out}")

    if not args.use_albert and not args.use_azure_openai and not args.use_gemini and not args.use_albert_python_cues and not args.use_azure_openai_python_cues and not args.calculate_python_cues and not args.use_foundry_agent_cue_discovery:
        if not args.use_local:
            return

    if args.use_azure_openai:
        azure_openai_result = run_azure_openai_raw_reasoner(
            raw_batch=raw_batch,
            tasks_config=tasks_context,
            clues_config=clues_config,
            deployment=args.azure_openai_deployment,
            api_version=args.azure_openai_api_version,
            system_prompt=raw_reasoner_prompt,
            sleep_seconds=args.azure_openai_sleep_seconds,
            timeout_seconds=args.azure_openai_timeout_seconds,
            chunk_rows=args.azure_openai_chunk_rows,
            chunk_overlap_rows=args.azure_openai_chunk_overlap_rows,
            max_chunks=args.azure_openai_max_chunks
        )
        args.azure_openai_out.parent.mkdir(parents=True, exist_ok=True)
        args.azure_openai_out.write_text(azure_openai_report_markdown(azure_openai_result), encoding="utf-8")
        print(f"Wrote {args.azure_openai_out}")

    if args.use_albert:
        albert_result = run_albert_raw_reasoner(
            raw_batch=raw_batch,
            tasks_config=tasks_context,
            clues_config=clues_config,
            model=args.albert_model,
            sleep_seconds=args.albert_sleep_seconds,
            chunk_rows=args.albert_chunk_rows,
            chunk_overlap_rows=args.albert_chunk_overlap_rows,
            max_chunks=args.albert_max_chunks,
            retry_count=args.albert_retry_count,
            retry_sleep_seconds=args.albert_retry_sleep_seconds,
            system_prompt=raw_reasoner_prompt
        )
        args.albert_out.parent.mkdir(parents=True, exist_ok=True)
        args.albert_out.write_text(albert_report_markdown(albert_result), encoding="utf-8")
        print(f"Wrote {args.albert_out}")

    if args.use_gemini:
        gemini_result = run_gemini_raw_reasoner(
            raw_batch=raw_batch,
            tasks_config=tasks_context,
            clues_config=clues_config,
            model=args.gemini_model,
            system_prompt=raw_reasoner_prompt,
            sleep_seconds=args.gemini_sleep_seconds,
            timeout_seconds=args.gemini_timeout_seconds
        )
        args.gemini_out.parent.mkdir(parents=True, exist_ok=True)
        args.gemini_out.write_text(gemini_report_markdown(gemini_result), encoding="utf-8")
        print(f"Wrote {args.gemini_out}")

    if args.use_local:
        local_result = run_local_raw_reasoner(
            raw_batch=raw_batch,
            tasks_config=tasks_context,
            clues_config=clues_config,
            model=args.local_model,
            base_url=args.local_base_url,
            sleep_seconds=args.local_sleep_seconds,
            chunk_rows=args.local_chunk_rows,
            chunk_overlap_rows=args.local_chunk_overlap_rows,
            max_chunks=args.local_max_chunks,
            timeout_seconds=args.local_timeout_seconds,
            system_prompt=raw_reasoner_prompt
        )
        args.local_out.parent.mkdir(parents=True, exist_ok=True)
        args.local_out.write_text(local_report_markdown(local_result), encoding="utf-8")
        print(f"Wrote {args.local_out}")

    if args.azure_openai_out.exists() and args.python_cues_md_out.exists():
        args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_out.write_text(
            build_comparison_report(
                azure_report_path=args.azure_openai_out,
                python_report_path=args.python_cues_md_out,
                foundry_report_path=args.foundry_comparison_report,
            ),
            encoding="utf-8"
        )
        print(f"Wrote {args.comparison_out}")


if __name__ == "__main__":
    main()
