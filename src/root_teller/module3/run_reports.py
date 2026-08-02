from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from root_teller.module2.llm import CachedJSONClient, load_api_settings

from .config import Module3Config, Module3Paths
from .reporting import generate_verified_report, report_markdown


def _emit(payload: Any) -> None:
    """Progress logging must never turn a completed case into a case failure."""
    try:
        print(
            payload if isinstance(payload, str) else json.dumps(payload),
            flush=True,
        )
    except OSError:
        pass


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def run(
    run_id: str,
    workers: int = 4,
    case_limit: int | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    config = Module3Config()
    paths = Module3Paths()
    output_root = paths.run_root / run_id
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"run directory is not empty: {output_root}")
    json_root = output_root / "reports_json"
    markdown_root = output_root / "reports_markdown"
    json_root.mkdir(parents=True, exist_ok=True)
    markdown_root.mkdir(parents=True, exist_ok=True)
    case_files = sorted((paths.module2_run / "cases").glob("*.json"))
    if case_limit is not None:
        case_files = case_files[:case_limit]

    client = None
    if not offline:
        client = CachedJSONClient(
            load_api_settings(paths.api_config),
            paths.response_cache,
            config.model,
            config.temperature,
            config.request_timeout_seconds,
            config.max_retries,
        )

    def execute(case_file: Path) -> tuple[str, dict[str, Any]]:
        incident_id = case_file.stem
        case = json.loads(case_file.read_text(encoding="utf-8"))
        rmg = json.loads(
            (paths.module2_run / "rmg" / case_file.name).read_text(encoding="utf-8")
        )
        result = generate_verified_report(
            case=case, rmg=rmg, client=client, config=config
        )
        return incident_id, result

    started = time.time()
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, path): path for path in case_files}
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                incident_id, result = future.result()
                results.append(result)
                (json_root / f"{incident_id}.json").write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
                (markdown_root / f"{incident_id}.md").write_text(
                    report_markdown(result), encoding="utf-8"
                )
                _emit(
                    {
                        "progress": f"{index}/{len(case_files)}",
                        "incident_id": incident_id,
                        "status": result["status"],
                        "rounds": len(result["rounds"]),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "incident_id": path.stem,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    results.sort(key=lambda item: item["incident_id"])
    verdicts = {
        verdict: sum(item["verifier_verdict"] == verdict for item in results)
        for verdict in ("PASS", "REVISE", "EVIDENCE_GAP")
    }
    fallback_count = sum(
        bool(round_payload["interpreter_call"].get("fallback"))
        + bool(round_payload["verifier_call"].get("fallback"))
        for item in results
        for round_payload in item["rounds"]
    )
    summary = {
        "schema_version": "module3-report-summary-1.0",
        "run_id": run_id,
        "config": config.to_dict(),
        "cases_requested": len(case_files),
        "cases_completed": len(results),
        "failures": failures,
        "verdicts": verdicts,
        "verified_report_rate": verdicts["PASS"] / len(results) if results else 0,
        "mean_verifier_rounds": (
            sum(len(item["rounds"]) for item in results) / len(results)
            if results
            else 0
        ),
        "fallback_calls": fallback_count,
        "llm_stats": client.stats if client else {},
        "elapsed_seconds": round(time.time() - started, 3),
        "ground_truth_in_report_context": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _emit(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)
    return summary


def main() -> None:
    args = _args()
    run(args.run_id, args.workers, args.case_limit, args.offline)


if __name__ == "__main__":
    main()
