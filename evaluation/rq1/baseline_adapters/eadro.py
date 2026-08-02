"""Functional EADRO reproduction on clean RCAEval RE2-OB.

This adapter deliberately keeps labels outside test-time feature extraction.
It ports the released EADRO architecture to the available PyTorch runtime
because the official DGL stack is not runnable on this workstation.
"""
from __future__ import annotations

import os

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tick.hawkes import HawkesADM4
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
ROOT = PROJECT / "baselines" / "eadro"
RAW = PROJECT / "dataset" / "RCAEval RE" / "RE2" / "RE2-OB" / "RE2-OB"
MANIFEST = Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT / "evaluation" / "rq1" / "manifests" / "active_split_manifest.csv"))
CACHE = ROOT / "cache" / "re2ob_clean_v1"
SERVICES = (
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
    "redis", "shippingservice",
)
SERVICE_INDEX = {name: idx for idx, name in enumerate(SERVICES)}
BIN_SECONDS = 60
METRIC_SUFFIXES = (
    "container-cpu-system-seconds-total", "container-cpu-usage-seconds-total",
    "container-cpu-user-seconds-total", "container-memory-usage-bytes",
    "container-memory-working-set-bytes", "container-network-receive-bytes-total",
    "container-network-transmit-bytes-total",
)
NUMBER = re.compile(r"(?<![A-Za-z])(?:[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z])")
HEX = re.compile(r"\b(?:0x)?[0-9a-fA-F]{8,}\b")
SPACE = re.compile(r"\s+")


def canonical_service(value: object) -> str:
    name = str(value).strip().lower().replace("_", "-")
    aliases = {"frontendservice": "frontend", "frontend-external": "frontend", "redis-cart": "redis"}
    name = aliases.get(name, name)
    if name.endswith("-service") and name.replace("-", "") in SERVICE_INDEX:
        name = name.replace("-", "")
    return name


def template(value: object) -> str:
    value = HEX.sub("<hex>", str(value).lower())
    return SPACE.sub(" ", NUMBER.sub("<num>", value)).strip()


@dataclass(frozen=True)
class Record:
    incident_id: str
    split: str
    root: str
    inject_time: float
    eligible: bool

    @property
    def directory(self) -> Path:
        return RAW / Path(self.incident_id)


def read_records() -> list[Record]:
    frame = pd.read_csv(MANIFEST)
    frame = frame.loc[frame.dataset_system.eq("RCAEval RE2-OB")]
    return [Record(str(row.incident_id), str(row.split), canonical_service(row.root_cause_service),
                   float(row.inject_time), bool(row.eligible)) for row in frame.itertuples(index=False)]


def split_records(records: list[Record], split: str) -> list[Record]:
    chosen = [r for r in records if r.split == split]
    if split == "test":
        chosen = [r for r in chosen if r.eligible]
    return chosen


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def time_grid(record: Record) -> tuple[int, int]:
    times = pd.read_csv(record.directory / "metrics.csv", usecols=["time"])["time"]
    times = pd.to_numeric(times, errors="coerce").dropna()
    start = int(math.floor(float(times.min())))
    bins = max(1, int(math.ceil((float(times.max()) - start + 1) / BIN_SECONDS)))
    return start, bins


def window_ids(times: np.ndarray, start: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    sec = np.floor(times - start).astype(np.int64)
    window = np.clip(sec // BIN_SECONDS, 0, bins - 1)
    within = np.clip(sec % BIN_SECONDS, 0, BIN_SECONDS - 1)
    return window, within


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    mean = float(np.nanmean(values))
    std = max(float(np.nanstd(values)), 1e-8)
    return np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)


def metric_tensor(record: Record, start: int, bins: int) -> np.ndarray:
    frame = pd.read_csv(record.directory / "metrics.csv", low_memory=False)
    timestamp = pd.to_numeric(frame["time"], errors="coerce").to_numpy(float)
    windows, offsets = window_ids(timestamp, start, bins)
    out = np.zeros((bins, len(SERVICES), BIN_SECONDS, len(METRIC_SUFFIXES)), dtype=np.float32)
    for service, service_idx in SERVICE_INDEX.items():
        for metric_idx, suffix in enumerate(METRIC_SUFFIXES):
            col = f"{service}_{suffix}"
            if col not in frame:
                continue
            values = zscore(pd.to_numeric(frame[col], errors="coerce").to_numpy(float))
            for w, t, value in zip(windows, offsets, values):
                out[w, service_idx, t, metric_idx] = value
    return out


def trace_tensor(record: Record, start: int, bins: int) -> np.ndarray:
    out = np.zeros((bins, len(SERVICES), BIN_SECONDS, 1), dtype=np.float32)
    path = record.directory / "traces.csv"
    if path.stat().st_size < 160:
        return out
    frame = pd.read_csv(path, usecols=["serviceName", "startTimeMillis", "duration"], dtype=str, low_memory=False)
    timestamp = pd.to_numeric(frame.startTimeMillis, errors="coerce").fillna(start * 1000).to_numpy(float) / 1000
    duration = pd.to_numeric(frame.duration, errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    windows, offsets = window_ids(timestamp, start, bins)
    services = frame.serviceName.map(canonical_service).to_numpy()
    sums = np.zeros_like(out)
    counts = np.zeros_like(out)
    for service, w, t, value in zip(services, windows, offsets, duration):
        if service in SERVICE_INDEX:
            sums[w, SERVICE_INDEX[service], t, 0] += value
            counts[w, SERVICE_INDEX[service], t, 0] += 1
    np.divide(sums, counts, out=out, where=counts > 0)
    for service_idx in range(len(SERVICES)):
        out[:, service_idx, :, 0] = zscore(out[:, service_idx, :, 0])
    return out


def vocabulary(records: list[Record], size: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        path = record.directory / "logs.csv"
        if path.stat().st_size < 120:
            continue
        clusters = json.loads((record.directory / "cluster_info.json").read_text(encoding="utf-8"))
        cluster_templates = {str(key): str(value["template"]) for key, value in clusters.items()}
        columns = pd.read_csv(path, nrows=0).columns
        usecols = ["cluster_id"] if "cluster_id" in columns else ["message"]
        for chunk in pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, chunksize=200000):
            values = (chunk.cluster_id.map(cluster_templates).fillna("Unseen") if "cluster_id" in chunk
                      else chunk.message.map(template))
            counts.update(values.tolist())
    return {item: index + 1 for index, (item, _) in enumerate(counts.most_common(size))}


def hawkes_baseline(events: list[np.ndarray], end_time: float) -> np.ndarray:
    if not any(len(item) for item in events):
        return np.zeros(len(events), dtype=np.float32)
    try:
        model = HawkesADM4(3.0)
        model.fit(events, end_time=end_time, baseline_start=np.ones(len(events)) * 0.2)
        return np.asarray(model.baseline, dtype=np.float32)
    except Exception:
        # A deterministic fallback is used only for numerically singular empty/near-empty Hawkes fits.
        return np.asarray([len(item) / max(end_time, 1.0) for item in events], dtype=np.float32)


def log_tensor(record: Record, start: int, bins: int, vocab: dict[str, int]) -> np.ndarray:
    dim = len(vocab) + 1
    out = np.zeros((bins, len(SERVICES), dim), dtype=np.float32)
    path = record.directory / "logs.csv"
    if path.stat().st_size < 120:
        return out
    columns = pd.read_csv(path, nrows=0).columns
    usecols = ["timestamp", "container_name", "cluster_id"] if "cluster_id" in columns else ["timestamp", "container_name", "message"]
    frame = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False)
    timestamp = pd.to_numeric(frame.timestamp, errors="coerce").fillna(start * 1e9).to_numpy(float) / 1e9
    services = frame.container_name.map(canonical_service).to_numpy()
    if "cluster_id" in frame:
        clusters = json.loads((record.directory / "cluster_info.json").read_text(encoding="utf-8"))
        cluster_templates = {str(key): str(value["template"]) for key, value in clusters.items()}
        events = frame.cluster_id.map(cluster_templates).fillna("Unseen")
    else:
        events = frame.message.map(template)
    event_ids = events.map(lambda x: vocab.get(x, 0)).to_numpy(int)
    windows = np.clip(np.floor((timestamp - start) / BIN_SECONDS).astype(int), 0, bins - 1)
    grouped: dict[tuple[int, int], list[list[float]]] = {}
    for service, event, ts, w in zip(services, event_ids, timestamp, windows):
        if service not in SERVICE_INDEX:
            continue
        key = (int(w), SERVICE_INDEX[service])
        if key not in grouped:
            grouped[key] = [[] for _ in range(dim)]
        grouped[key][int(event)].append(float(ts - (start + int(w) * BIN_SECONDS)))
    for (w, service), values in grouped.items():
        knots = [np.asarray(sorted(x), dtype=float) + np.arange(len(x)) * 1e-5 for x in values]
        out[w, service] = hawkes_baseline(knots, float(BIN_SECONDS))
    return out


def historic_edges(records: list[Record]) -> list[tuple[int, int]]:
    seen: dict[tuple[str, str], str] = {}
    pending: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    edges: set[tuple[int, int]] = set()
    for record in records:
        path = record.directory / "traces.csv"
        if path.stat().st_size < 160:
            continue
        for chunk in pd.read_csv(path, usecols=["traceID", "spanID", "serviceName", "parentSpanID"], dtype=str, keep_default_na=False, chunksize=200000):
            for trace, span, service, parent in zip(chunk.traceID, chunk.spanID, chunk.serviceName.map(canonical_service), chunk.parentSpanID):
                if service not in SERVICE_INDEX:
                    continue
                key = (str(trace), str(span))
                for child in pending.pop(key, []):
                    if child != service:
                        edges.add((SERVICE_INDEX[service], SERVICE_INDEX[child]))
                seen[key] = service
                if parent:
                    parent_key = (str(trace), str(parent))
                    if parent_key in seen and seen[parent_key] != service:
                        edges.add((SERVICE_INDEX[seen[parent_key]], SERVICE_INDEX[service]))
                    else:
                        pending[parent_key].append(service)
    return sorted(edges)


def cache_file(record: Record, vocab: dict[str, int]) -> Path:
    identity = record.incident_id + "|" + json.dumps(vocab, sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return CACHE / f"{digest}.npz"


def build_case(record: Record, vocab: dict[str, int], rebuild: bool) -> dict[str, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_file(record, vocab)
    if path.exists() and not rebuild:
        data = np.load(path, allow_pickle=False)
        return {key: data[key] for key in data.files}
    start, bins = time_grid(record)
    metrics = metric_tensor(record, start, bins)
    traces = trace_tensor(record, start, bins)
    logs = log_tensor(record, start, bins, vocab)
    starts = start + np.arange(bins) * BIN_SECONDS
    labels = np.where(starts + BIN_SECONDS * 0.5 >= record.inject_time, SERVICE_INDEX[record.root], -1).astype(np.int64)
    payload = {"metrics": metrics, "traces": traces, "logs": logs, "labels": labels,
               "starts": starts.astype(np.int64)}
    np.savez_compressed(path, **payload)
    return payload


class WindowDataset(Dataset):
    def __init__(self, cases: list[tuple[Record, dict[str, np.ndarray]]]) -> None:
        self.items: list[dict[str, object]] = []
        for record, case in cases:
            for index in range(case["metrics"].shape[0]):
                self.items.append({"incident": record.incident_id, "metrics": case["metrics"][index],
                                   "traces": case["traces"][index], "logs": case["logs"][index],
                                   "label": int(case["labels"][index])})
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, object]: return self.items[index]


def collate(items: list[dict[str, object]]) -> dict[str, object]:
    return {"metrics": torch.tensor(np.stack([x["metrics"] for x in items]), dtype=torch.float32),
            "traces": torch.tensor(np.stack([x["traces"] for x in items]), dtype=torch.float32),
            "logs": torch.tensor(np.stack([x["logs"] for x in items]), dtype=torch.float32),
            "labels": torch.tensor([x["label"] for x in items], dtype=torch.long),
            "incidents": [str(x["incident"]) for x in items]}


class CausalConv(nn.Module):
    def __init__(self, channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, hidden, 2, padding=1), nn.BatchNorm1d(hidden), nn.ReLU())
        self.attn = nn.Linear(hidden, 1)
    def forward(self, x: Tensor) -> Tensor:
        states = self.net(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        weights = torch.softmax(self.attn(states).squeeze(-1), dim=1)
        return torch.sum(states * weights.unsqueeze(-1), dim=1)


class GATv2Compat(nn.Module):
    def __init__(self, dimension: int, heads: int, edges: list[tuple[int, int]]) -> None:
        super().__init__()
        self.heads, self.dimension = heads, dimension
        self.left = nn.Linear(dimension, dimension * heads, bias=False)
        self.right = nn.Linear(dimension, dimension * heads, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, dimension))
        nn.init.xavier_uniform_(self.attn)
        adjacency = torch.zeros(len(SERVICES), len(SERVICES), dtype=torch.bool)
        for source, target in edges:
            adjacency[target, source] = True
        adjacency.fill_diagonal_(True)
        self.register_buffer("adjacency", adjacency)
    def forward(self, x: Tensor) -> Tensor:
        left = self.left(x).view(x.shape[0], x.shape[1], self.heads, self.dimension)
        right = self.right(x).view_as(left)
        pair = F.leaky_relu(left.unsqueeze(2) + right.unsqueeze(1), 0.2)
        scores = torch.einsum("btshe,he->btsh", pair, self.attn).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~self.adjacency.unsqueeze(0).unsqueeze(0), -1e9)
        weights = torch.softmax(scores, dim=-1)
        values = left.permute(0, 2, 1, 3)
        out = torch.einsum("bhij,bhje->bhie", weights, values).permute(0, 2, 1, 3)
        return out.max(dim=2).values


class EadroCompat(nn.Module):
    def __init__(self, events: int, edges: list[tuple[int, int]]) -> None:
        super().__init__()
        self.metric = CausalConv(7, 64)
        self.trace = CausalConv(1, 64)
        self.log = nn.Linear(events, 16)
        self.fuse = nn.Linear(144, 128)
        self.graph = GATv2Compat(64, 4, edges)
        self.pool_score = nn.Linear(64, 1)
        self.detector = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        self.locator = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, len(SERVICES)))
    def forward(self, metrics: Tensor, traces: Tensor, logs: Tensor) -> tuple[Tensor, Tensor]:
        batch, nodes = metrics.shape[:2]
        metric = self.metric(metrics.reshape(batch * nodes, BIN_SECONDS, 7)).reshape(batch, nodes, -1)
        trace = self.trace(traces.reshape(batch * nodes, BIN_SECONDS, 1)).reshape(batch, nodes, -1)
        local = torch.cat((trace, self.log(logs), metric), dim=-1)
        local = F.glu(self.fuse(local), dim=-1)
        node = self.graph(local)
        weights = torch.softmax(self.pool_score(node).squeeze(-1), dim=1)
        status = torch.sum(node * weights.unsqueeze(-1), dim=1)
        return self.detector(status), self.locator(status)


def score(model: EadroCompat, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, list[float]]]:
    model.eval(); rankings: defaultdict[str, list[np.ndarray]] = defaultdict(list); weights: defaultdict[str, list[float]] = defaultdict(list)
    correct = {1: 0, 5: 0}; total = 0
    with torch.no_grad():
        for batch in loader:
            detect, locate = model(batch["metrics"].to(device), batch["traces"].to(device), batch["logs"].to(device))
            loc = torch.softmax(locate, dim=1).cpu().numpy(); anomaly = torch.softmax(detect, dim=1)[:, 1].cpu().numpy()
            labels = batch["labels"].numpy()
            for incident, pred, weight, label in zip(batch["incidents"], loc, anomaly, labels):
                rankings[incident].append(pred); weights[incident].append(float(weight))
                if label >= 0:
                    rank = int(np.argsort(-pred).tolist().index(int(label)))
                    correct[1] += int(rank == 0); correct[5] += int(rank < 5); total += 1
    incident_scores: dict[str, list[float]] = {}
    for incident, predictions in rankings.items():
        current_weights = np.asarray(weights[incident], dtype=np.float64)
        if current_weights.sum() <= 1e-9: current_weights = np.ones_like(current_weights)
        aggregate = np.average(np.stack(predictions), axis=0, weights=current_weights)
        incident_scores[incident] = aggregate.tolist()
    return {"window_A@1": correct[1] / max(total, 1), "window_A@5": correct[5] / max(total, 1), "windows": total}, incident_scores


def incident_metrics(scores: dict[str, list[float]], records: list[Record]) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows=[]; hits=np.zeros(5, dtype=float)
    for record in records:
        values = np.asarray(scores[record.incident_id]); order = np.argsort(-values); truth = SERVICE_INDEX[record.root]
        rank = int(np.where(order == truth)[0][0]); hits += np.asarray([rank < k for k in range(1, 6)], float)
        rows.append({"incident_id": hashlib.sha256(record.incident_id.encode()).hexdigest()[:16],
                     "ranking": [SERVICES[i] for i in order[:5]], "root_cause_service": record.root, "rank": rank + 1})
    hits /= max(len(records), 1)
    return {"A@1": float(hits[0]), "A@5": float(hits[4]), "Avg@5": float(hits.mean()), "cases": len(records)}, rows


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "frozen_test"), default="development")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-vocab-size", type=int, default=256); parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--run-id", default="development_default")
    args=parser.parse_args(); seed_everything(args.seed); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records=read_records(); train=split_records(records,"train"); validation=split_records(records,"validation"); test=split_records(records,"test")
    vocab=vocabulary(train, args.log_vocab_size)
    selected = train + validation + (test if args.phase == "frozen_test" else [])
    prepared={r.incident_id: build_case(r,vocab,args.rebuild_cache) for r in selected}
    edges=historic_edges(train if args.phase == "development" else train + validation)
    train_cases=[(r,prepared[r.incident_id]) for r in train] if args.phase == "development" else [(r,prepared[r.incident_id]) for r in train+validation]
    val_cases=[(r,prepared[r.incident_id]) for r in validation]
    test_cases=[(r,prepared[r.incident_id]) for r in test]
    train_loader=DataLoader(WindowDataset(train_cases), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader=DataLoader(WindowDataset(val_cases), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader=DataLoader(WindowDataset(test_cases), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model=EadroCompat(len(vocab)+1,edges).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best_state=None; best=-1.0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            labels=batch["labels"].to(device); detect, locate=model(batch["metrics"].to(device),batch["traces"].to(device),batch["logs"].to(device))
            detect_labels=(labels>=0).long(); loc_loss=F.cross_entropy(locate,labels,ignore_index=-1); loss=0.5*F.cross_entropy(detect,detect_labels)+0.5*loc_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.item()))
        if args.phase == "development":
            window, scores=score(model,val_loader,device); metrics,_=incident_metrics(scores,validation)
            history.append({"epoch":epoch,"loss":float(np.mean(losses)),**window,**metrics})
            if metrics["A@1"] > best:
                best=metrics["A@1"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            history.append({"epoch":epoch,"loss":float(np.mean(losses))})
    if args.phase == "development":
        model.load_state_dict(best_state); window,scores=score(model,val_loader,device); metrics,rows=incident_metrics(scores,validation)
    else:
        window,scores=score(model,test_loader,device); metrics,rows=incident_metrics(scores,test)
    output=ROOT/"runs"/args.phase/args.run_id; output.mkdir(parents=True,exist_ok=True)
    torch.save({"state":model.state_dict(),"vocab":vocab,"edges":edges,"args":vars(args)},output/"best_validation.pt")
    prefix="validation" if args.phase == "development" else "test"
    (output/f"{prefix}_summary.json").write_text(json.dumps({"metrics":metrics,"window":window,"history":history,"edges":edges},indent=2)+"\n",encoding="utf-8")
    (output/f"{prefix}_predictions_private.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"metrics":metrics,"window":window,"device":str(device),"edges":len(edges)},indent=2))

if __name__ == "__main__": main()
