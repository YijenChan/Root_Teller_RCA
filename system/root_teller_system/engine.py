from __future__ import annotations

import hashlib
import csv
import json
import os
import shutil
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from root_teller.module1.config import FeatureConfig
from root_teller.module1.data import ReferenceStats, to_torch_case
from root_teller.module1.features import CaseSpec, extract_case, load_case, save_case
from root_teller.module1.model import ModelConfig, PerceptionRCA
from root_teller.module2 import window_export
from root_teller.module2.agents import EvidenceSteward, WindowInvestigator
from root_teller.module2.config import Module2Config
from root_teller.module2.contracts import stable_id
from root_teller.module2.llm import CachedJSONClient, load_api_settings
from root_teller.module2.run import run_case_blind, run_case_default
from root_teller.module3.config import Module3Config
from root_teller.module3.feedback import FeedbackRMGOverlay
from root_teller.module3.reporting import generate_verified_report
from root_teller.multidataset import rq1_sn, rq1_tt
from root_teller.multidataset.rq1_protocol_v2 import configure_services


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(
    os.environ.get("ROOTTELLER_WORKSPACE", str(SYSTEM_ROOT.parent))
).resolve()
RUNTIME_ROOT = SYSTEM_ROOT / "runtime"
CHECKPOINT_ROOT = SYSTEM_ROOT / "checkpoints"


class LocalPaths:
    def __init__(self, clean_root: Path) -> None:
        self.clean_root = clean_root
        self.corrupted_root = WORKSPACE / "dataset" / "dataset_corrupted" / "_unused"
        self.cache_root = RUNTIME_ROOT / "feature_cache"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _opaque(value: str) -> str:
    return "case-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _is_re2_incident(path: Path) -> bool:
    return all((path / name).exists() for name in ("metrics.csv", "logs.csv", "traces.csv"))


def _dataset_from_path(path: Path) -> str | None:
    lowered = str(path).lower()
    if "re2-ob" in lowered:
        return "re2_ob"
    if "re2-tt" in lowered:
        return "re2_tt"
    if "eadro-sn" in lowered or "sn dataset" in lowered or path.name.startswith("SN."):
        return "eadro_sn"
    search_root = path if path.is_dir() else path.parent
    if any(search_root.rglob("SN.fault-*.json")) or (
        (search_root / "metrics").is_dir() and (search_root / "logs.json").exists()
    ):
        return "eadro_sn"
    metric = next(
        (
            item for item in search_root.rglob("metrics.csv")
            if (item.parent / "logs.csv").exists() and (item.parent / "traces.csv").exists()
        ),
        None,
    )
    if metric is not None:
        header = metric.open("r", encoding="utf-8-sig", errors="ignore").readline().lower()
        return "re2_tt" if "ts-" in header or "ts_" in header else "re2_ob"
    return None


def inspect_case_path(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path.strip().strip('"')).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    dataset = _dataset_from_path(path)
    variants: list[dict[str, Any]] = []
    if dataset in {"re2_ob", "re2_tt"}:
        if _is_re2_incident(path):
            candidates = [path]
        else:
            family = path
            candidates = [
                item for item in sorted(family.iterdir())
                if item.is_dir() and item.name.isdigit() and _is_re2_incident(item)
            ]
            if not candidates:
                candidates = [
                    item for item in sorted(path.glob("*/*"))
                    if item.is_dir() and item.name.isdigit() and _is_re2_incident(item)
                ]
        for item in candidates:
            family = item.parent
            inject_time = None
            inject = item / "inject_time.txt"
            if inject.exists():
                try:
                    inject_time = float(inject.read_text(encoding="utf-8").strip())
                except ValueError:
                    pass
            variants.append(
                {
                    "id": f"{family.name}/{item.name}",
                    "label": f"Injection {item.name}",
                    "path": str(item),
                    "family": family.name,
                    "repetition": item.name,
                    "inject_time": inject_time,
                }
            )
    elif dataset == "eadro_sn":
        data_root = path
        while data_root != data_root.parent and not list(data_root.glob("SN.fault-*.json")):
            data_root = data_root.parent
        annotations = sorted(data_root.glob("SN.fault-*.json"))
        if path.is_file() and path.name.startswith("SN.fault-"):
            annotations = [path]
        elif path.is_dir() and path.name.startswith("SN.") and not path.name.startswith("SN.fault-"):
            stamp = path.name.removeprefix("SN.")
            annotations = [data_root / f"SN.fault-{stamp}.json"]
        for capture_index, annotation in enumerate(sorted(data_root.glob("SN.fault-*.json"))):
            if annotation not in annotations:
                continue
            stamp = annotation.name.removeprefix("SN.fault-").removesuffix(".json")
            capture = data_root / f"SN.{stamp}"
            metadata = _json(annotation)
            for fault_index, fault in enumerate(metadata.get("faults", [])):
                variants.append(
                    {
                        "id": f"sn-c{capture_index:02d}-f{fault_index:02d}",
                        "label": f"{fault.get('name', 'unknown')} / {fault.get('fault', 'fault')}",
                        "path": str(capture),
                        "annotation": str(annotation),
                        "capture_index": capture_index,
                        "fault_index": fault_index,
                        "inject_time": fault.get("start"),
                    }
                )
    else:
        raise ValueError("Unrecognized case layout. Select an RE2-OB, RE2-TT, or Eadro-SN case.")
    if not variants:
        raise ValueError("No runnable incident was found under the selected path.")
    return {
        "dataset": dataset,
        "dataset_label": {"re2_ob": "RCAEval RE2-OB", "re2_tt": "RCAEval RE2-TT", "eadro_sn": "Eadro-SN"}[dataset],
        "source_path": str(path),
        "variants": variants,
    }


def safe_extract_zip(source: Path, destination: Path, max_uncompressed: int = 3_000_000_000) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            total += member.file_size
            if total > max_uncompressed:
                raise ValueError("Uploaded archive exceeds the uncompressed size limit.")
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError("Unsafe archive path detected.")
        archive.extractall(destination)
    children = [item for item in destination.iterdir() if item.name != source.name]
    return children[0] if len(children) == 1 and children[0].is_dir() else destination


def _manifest_record(dataset: str, incident_id: str) -> dict[str, Any] | None:
    candidates = [
        CHECKPOINT_ROOT / dataset / "fold_manifest.json",
        WORKSPACE / "runs" / "rq1_protocol_v2_seed42" / dataset / "fold_manifest.json",
    ]
    manifest = next((item for item in candidates if item.exists()), None)
    if manifest is None:
        return None
    records = _json(manifest)
    matches = [row for row in records if row.get("incident_id") == incident_id]
    test = [row for row in matches if row.get("role") == "test"]
    return (test or matches or [None])[0]


def _checkpoint(dataset: str, incident_id: str, capture_index: int | None = None) -> Path:
    if dataset == "eadro_sn":
        fold = int(capture_index or 0)
        candidates = [
            CHECKPOINT_ROOT / dataset / f"fold_{fold}.pt",
            WORKSPACE / "runs" / "rq1_root_teller_three_seed" / "seed_42" / dataset / "module1" / f"fold_{fold}" / "checkpoint.pt",
        ]
    else:
        record = _manifest_record(dataset, incident_id)
        fold = int(record["outer_fold"]) if record else 0
        candidates = [
            CHECKPOINT_ROOT / dataset / f"fold_{fold}.pt",
            WORKSPACE / "runs" / "rq1_protocol_v2_seed42" / dataset / f"fold_{fold}" / "checkpoint.pt",
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint is available for {dataset} fold {fold}.")


def _matching_cache(dataset: str, incident_id: str) -> Path | None:
    if dataset == "re2_ob":
        root = WORKSPACE / "cache" / "module1_re2ob"
        candidates = list(root.glob(f"*/**/CLEAN/{incident_id}.npz"))
        valid = []
        for item in candidates:
            try:
                case = load_case(item)
                if int(case["log_x"].shape[2]) == 387:
                    valid.append(item)
            except Exception:
                continue
        return max(valid, key=lambda item: item.stat().st_size) if valid else None
    if dataset == "re2_tt":
        candidates = list((WORKSPACE / "cache" / "rq1_multidataset_clean" / "re2_tt" / "features").glob(f"*/**/{incident_id}.npz"))
        return max(candidates, key=lambda item: item.stat().st_size) if candidates else None
    return None


def _load_re2_case(dataset: str, variant: dict[str, Any], progress: Callable[[str, int], None]) -> dict[str, Any]:
    incident_id = variant["id"]
    cached = _matching_cache(dataset, incident_id)
    if cached:
        progress("Loading cached multimodal features", 18)
        case = load_case(cached)
        case["incident_id"] = incident_id
        return case
    progress("Extracting metric, log, and trace features", 12)
    incident_path = Path(variant["path"])
    dataset_root = incident_path.parent.parent
    if dataset == "re2_tt":
        with (incident_path / "metrics.csv").open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            columns = next(csv.reader(handle))
        candidates = set()
        for column in columns:
            if column == "time" or "_" not in column:
                continue
            service = column.split("_", 1)[0].strip().lower()
            if not service.startswith("gke-") and service not in {"loadgenerator", "ts"}:
                candidates.add(service)
        services = tuple(sorted(candidates))
        rq1_tt.configure_services(services)
    inject_time = variant.get("inject_time")
    if inject_time is None:
        inject_time = float((incident_path / "inject_time.txt").read_text(encoding="utf-8").strip())
    family, fault_type = variant["family"].rsplit("_", 1)
    spec = CaseSpec(incident_id, "interactive", True, family, fault_type, float(inject_time))
    case = extract_case(LocalPaths(dataset_root), spec, "CLEAN", FeatureConfig(log_backend="sbert"))
    destination = RUNTIME_ROOT / "feature_cache" / dataset / f"{incident_id}.npz"
    save_case(destination, case)
    return case


def _load_sn_case(variant: dict[str, Any], progress: Callable[[str, int], None]) -> dict[str, Any]:
    index = int(variant["capture_index"])
    fault_index = int(variant["fault_index"])
    cache = WORKSPACE / "cache" / "rq1_multidataset_clean" / "eadro_sn" / "cases" / f"c{index:02d}_f{fault_index:02d}.pt"
    if cache.exists():
        progress("Loading cached multimodal features", 18)
        return torch.load(cache, map_location="cpu", weights_only=False)
    progress("Extracting Eadro-SN capture features", 12)
    capture = Path(variant["path"])
    metadata = _json(Path(variant["annotation"]))
    faults = metadata["faults"]
    end = float(faults[fault_index + 1]["start"]) if fault_index + 1 < len(faults) else float(metadata["end"])
    services = tuple(sorted(rq1_sn.canonical(path.stem) for path in (capture / "metrics").glob("*.csv")))
    return rq1_sn.build_case(capture, faults[fault_index], end, index, fault_index, services)


def _export_packs(case: dict[str, Any], checkpoint_path: Path, pack_root: Path, progress: Callable[[str, int], None]) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    services = tuple(checkpoint["services"])
    configure_services(services)
    rq1_sn.configure(services) if str(case["incident_id"]).startswith("sn-") else None
    config = ModelConfig(**checkpoint["model_config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PerceptionRCA(
        metric_dim=int(case["metric_x"].shape[2]),
        log_dim=int(case["log_x"].shape[2]),
        trace_dim=int(case["trace_x"].shape[2]),
        config=config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    reference = ReferenceStats.from_state_dict(checkpoint["reference"])
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    pack_root.mkdir(parents=True, exist_ok=True)
    bins = int(case["metric_x"].shape[1])
    for activation_order, bin_index in enumerate(range(bins - 1, -1, -1)):
        sliced = window_export._slice_case(case, bin_index)
        tensor_case = to_torch_case(sliced, reference, device)
        with torch.no_grad():
            output = model(tensor_case)
        pack = window_export._pack(tensor_case, output, digest, f"W{activation_order:02d}", bin_index)
        (pack_root / f"W{activation_order:02d}.json").write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
        progress(f"Perception Agent encoded window {activation_order + 1}/{bins}", 20 + int(35 * (activation_order + 1) / bins))
    return checkpoint, {"device": str(device), "windows": bins, "services": list(services)}


def _client(job_root: Path, live_llm: bool, config: Module2Config) -> CachedJSONClient | None:
    if not live_llm:
        return None
    config_path = WORKSPACE / "config" / "API_KEY.txt"
    has_environment = bool(
        (os.environ.get("ROOTTELLER_API_KEY") or os.environ.get("OPENAI_API_KEY"))
        and (
            os.environ.get("ROOTTELLER_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
        )
    )
    if not has_environment and not config_path.exists():
        return None
    return CachedJSONClient(
        load_api_settings(config_path), job_root / "llm_cache", config.model,
        config.temperature, config.request_timeout_seconds, config.max_retries,
    )


def _ranking_scores(result: dict[str, Any], protocol: str) -> dict[str, float]:
    if protocol == "blind":
        state = result["actual_stop"]["diagnosis_state"]
        return {str(k): float(v) for k, v in state["fused_scores"].items()}
    cycles = result["hierarchical_rca_loop"]
    if cycles:
        return {str(k): float(v) for k, v in cycles[-1]["steward_after_inspection"]["fused_scores"].items()}
    ranking = result["default_exhaustive"]["ranking"]
    return {entity: 1.0 / (index + 1) for index, entity in enumerate(ranking)}


def _report_input(result: dict[str, Any], protocol: str) -> dict[str, Any]:
    if protocol == "default":
        return result
    clone = dict(result)
    clone["default_exhaustive"] = {"ranking": result["actual_stop"]["ranking"]}
    clone["snapshot"] = dict(result["snapshot"])
    clone["snapshot"].setdefault("unresolved_issues", [])
    return clone


def _timeline(rmg: dict[str, Any], result: dict[str, Any], protocol: str) -> list[dict[str, Any]]:
    activated = set(rmg["windows"])
    rows = []
    for pack in sorted(rmg["windows"].values(), key=lambda item: item["window"]["activation_order"]):
        leader = max(pack["ranked_candidates"], key=lambda item: item["anomaly_score"])
        rows.append({
            "window_id": pack["window"]["window_id"],
            "activation_order": pack["window"]["activation_order"],
            "source_bin": pack["window"].get("source_bin"),
            "anomaly": round(float(leader["anomaly_score"]), 4),
            "leader": leader["entity_id"],
            "activated": pack["window"]["window_id"] in activated,
        })
    if protocol == "blind":
        available = result["actual_stop"]["available_windows"]
        for index in range(len(rows), available):
            rows.append({"window_id": f"W{index:02d}", "activation_order": index, "source_bin": None, "anomaly": None, "leader": None, "activated": False})
    return rows


def diagnose(
    *, job_id: str, dataset: str, variant: dict[str, Any], protocol: str,
    live_llm: bool, progress: Callable[[str, int], None],
) -> dict[str, Any]:
    started = time.time()
    job_root = RUNTIME_ROOT / "jobs" / job_id
    job_root.mkdir(parents=True, exist_ok=True)
    progress("Validating case layout", 5)
    case = _load_sn_case(variant, progress) if dataset == "eadro_sn" else _load_re2_case(dataset, variant, progress)
    incident_id = str(case["incident_id"])
    checkpoint_path = _checkpoint(dataset, incident_id, variant.get("capture_index"))
    _, runtime = _export_packs(case, checkpoint_path, job_root / "packs", progress)
    config = Module2Config()
    client = _client(job_root, live_llm, config)
    steward = EvidenceSteward(client, config) if client else None
    investigator = WindowInvestigator(client, config) if client else None
    progress("Running hierarchical agent collaboration", 62)
    if protocol == "blind":
        result, rmg = run_case_blind(job_root / "packs", config, steward, investigator)
    else:
        result, rmg = run_case_default(job_root / "packs", config, steward, investigator)
    progress("Generating and verifying RCA report", 86)
    report = generate_verified_report(case=_report_input(result, protocol), rmg=rmg, client=client, config=Module3Config())
    scores = _ranking_scores(result, protocol)
    overlay = FeedbackRMGOverlay(result["incident_id"], scores)
    payload = {
        "schema_version": "root-teller-system-result-1.0",
        "job_id": job_id,
        "dataset": dataset,
        "dataset_label": {"re2_ob": "RCAEval RE2-OB", "re2_tt": "RCAEval RE2-TT", "eadro_sn": "Eadro-SN"}[dataset],
        "source": {"path": variant["path"], "selection": variant["id"], "opaque_case_id": _opaque(incident_id)},
        "protocol": protocol,
        "live_llm": bool(client),
        "model": config.model if client else "deterministic fallback",
        "runtime": {**runtime, "elapsed_seconds": round(time.time() - started, 3)},
        "result": result,
        "rmg": rmg,
        "report": report,
        "timeline": _timeline(rmg, result, protocol),
        "ranking": overlay.ranking(decay=0.90),
        "feedback": overlay.artifact(decay=0.90),
        "llm_stats": client.stats if client else {"offline": True},
    }
    (job_root / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    progress("Diagnosis complete", 100)
    return payload


def apply_feedback(payload: dict[str, Any], entity: str, verdict: str, message: str) -> dict[str, Any]:
    feedback = payload.get("feedback", {})
    base_scores = {row["entity_id"]: float(row["base_score"]) for row in feedback.get("current_ranking", payload["ranking"])}
    overlay = FeedbackRMGOverlay(payload["result"]["incident_id"], base_scores)
    overlay.reject_counts.update({str(k): int(v) for k, v in feedback.get("reject_counts", {}).items()})
    overlay.feedback_events.extend(feedback.get("feedback_events", []))
    overlay.confirmed_hypothesis_id = feedback.get("confirmed_hypothesis_id")
    if entity not in base_scores:
        raise ValueError("Feedback can only target a current RMG hypothesis.")
    event = overlay.commit(entity=entity, verdict=verdict, round_index=len(overlay.feedback_events) + 1)
    event["operator_message"] = message[:1200]
    overlay.feedback_events[-1]["operator_message"] = message[:1200]
    payload["feedback"] = overlay.artifact(decay=0.90)
    payload["ranking"] = payload["feedback"]["current_ranking"]
    payload["report"]["report"]["root_cause_summary"] = (
        f"{payload['ranking'][0]['entity_id']} is the leading evidence-grounded root-cause hypothesis after structured SRE feedback."
    )
    payload["report"]["report"]["ranked_alternatives"] = [row["entity_id"] for row in payload["ranking"][1:5]]
    job_root = RUNTIME_ROOT / "jobs" / payload["job_id"]
    (job_root / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {"event": event, "feedback": payload["feedback"], "ranking": payload["ranking"], "report": payload["report"]}


def copy_upload(upload: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(upload, target)
    return target
