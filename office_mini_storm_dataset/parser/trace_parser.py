# trace_parser.py (anonymized & open-source safe version)
# Converts JSON traces (e.g., from Fluent Bit or OpenTelemetry) into graph features for RCA models.

import json
import numpy as np
import pickle
from typing import List, Dict
import os
import re
from config import TRACE_PARSER_CONFIG


class TraceToGraphConverter:

    def __init__(self):
        self.http_methods = TRACE_PARSER_CONFIG["http_methods"]
        self.status_codes = TRACE_PARSER_CONFIG["status_codes"]
        self.db_operations = TRACE_PARSER_CONFIG["db_operations"]
        self.services = TRACE_PARSER_CONFIG["services"]
        self.trace_paths = TRACE_PARSER_CONFIG["trace_paths"]
        self.suspicious_patterns = TRACE_PARSER_CONFIG["suspicious_patterns"]

    # -------------------------
    # Main conversion routine
    # -------------------------
    def parse_trace_to_graph(self, trace_data: Dict) -> Dict:
        timestamp = trace_data.get("timestamp", 0)
        trace = trace_data.get("trace", trace_data)
        trace_id = trace.get("trace_id", "")
        spans = trace.get("spans", [])

        # Build index mapping
        span_id_to_idx, span_ids = {}, []
        for idx, span in enumerate(spans):
            span_id = span.get("context", {}).get("span_id", f"anon_{idx}")
            span_id_to_idx[span_id] = idx
            span_ids.append(span_id)

        # Node features
        node_features = np.array(
            [self._extract_span_features(span) for span in spans],
            dtype=np.float32
        )

        # Edge list (bidirectional)
        edge_list = []
        for idx, span in enumerate(spans):
            parent_id = span.get("parent_id")
            if parent_id and parent_id in span_id_to_idx:
                parent_idx = span_id_to_idx[parent_id]
                edge_list.append([parent_idx, idx])
                edge_list.append([idx, parent_idx])

        edge_index = (
            np.array(edge_list, dtype=np.int64).T
            if edge_list else np.empty((2, 0), dtype=np.int64)
        )

        # Label assignment
        label = self._classify_trace(trace_data)

        return {
            "timestamp": timestamp,
            "trace_id": trace_id,
            "node_features": node_features,
            "edge_index": edge_index,
            "span_ids": span_ids,
            "label": label,
            "num_nodes": len(spans)
        }

    # -------------------------
    # Span feature extraction
    # -------------------------
    def _extract_span_features(self, span: Dict) -> List[float]:
        features = []
        attrs = span.get("attributes", {})

        # Remove sensitive keys before processing
        sensitive_keys = {"user", "username", "email", "ip", "host"}
        for k in list(attrs.keys()):
            if any(sk in k.lower() for sk in sensitive_keys):
                attrs[k] = "anonymized"

        # 1) HTTP method one-hot
        http_method = attrs.get("http.method", "")
        features.extend([1.0 if m == http_method else 0.0 for m in self.http_methods])

        # 2) Status code one-hot
        status_code = span.get("status_code", "STATUS_CODE_UNSET")
        features.extend([1.0 if s == status_code else 0.0 for s in self.status_codes])

        # 3) Service name one-hot
        service_name = attrs.get("service.name", "")
        features.extend([1.0 if s == service_name else 0.0 for s in self.services])

        # 4) DB operation one-hot
        db_op = attrs.get("db.operation", "")
        features.extend([1.0 if op == db_op else 0.0 for op in self.db_operations])

        # 5) Duration normalized
        start_time = span.get("start_time", 0)
        end_time = span.get("end_time", 0)
        duration = max(0, end_time - start_time)
        features.append(min(1.0, duration / 10000.0))

        # 6) HTTP status normalization
        http_status = attrs.get("http.status_code", 200)
        features.append((http_status - 200) / 300.0)

        # 7) Auth attempt normalization
        auth_attempt = attrs.get("auth.attempt", 0)
        features.append(min(1.0, auth_attempt / 100.0))

        # 8) HTTP target hashed index (path anonymized)
        http_target = attrs.get("http.target", "")
        path_hash = (abs(hash(http_target)) % 1000) / 1000.0
        features.append(path_hash)

        # 9) Suspicious query detection (text only)
        query_string = attrs.get("http.query_string", "")
        has_suspicious_query = 0.0
        for pattern in self.suspicious_patterns:
            if re.search(pattern, query_string, re.IGNORECASE):
                has_suspicious_query = 1.0
                break
        features.append(has_suspicious_query)

        # 10) Request rate normalization
        request_rate = attrs.get("request.rate", 0)
        features.append(min(1.0, request_rate / 1000.0))

        return features

    # -------------------------
    # Classification (neutralized)
    # -------------------------
    def _classify_trace(self, trace_data: Dict) -> int:
        """Heuristic classification; anonymized categories."""
        trace = trace_data.get("trace", trace_data)
        spans = trace.get("spans", [])

        error_count = sum(1 for s in spans if s.get("status_code") == "STATUS_CODE_ERROR")
        auth_failures = sum(1 for s in spans if s.get("attributes", {}).get("auth.attempt", 0) > 10)
        suspicious_query = any(
            any(re.search(p, s.get("attributes", {}).get("http.query_string", ""), re.IGNORECASE)
                for p in self.suspicious_patterns)
            for s in spans
        )
        high_rate = any(s.get("attributes", {}).get("request.rate", 0) > 100 for s in spans)

        # Return generic numeric category
        if high_rate and error_count >= 1:
            return 3
        if auth_failures > 0:
            return 2
        if suspicious_query:
            return 1
        return 0

    # -------------------------
    # File I/O
    # -------------------------
    def load_traces_from_file(self, file_path: str) -> List[Dict]:
        traces = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    print("[WARN] Skipped invalid JSON line")
        return traces

    def convert_traces_to_graphs(self, input_file: str, output_pkl: str, output_npy: str = None):
        traces = self.load_traces_from_file(input_file)
        if not traces:
            print("[WARN] No valid trace data found.")
            return None

        graphs = []
        for trace_data in traces:
            try:
                graphs.append(self.parse_trace_to_graph(trace_data))
            except Exception as e:
                print(f"[WARN] Failed to parse trace: {e}")

        with open(output_pkl, "wb") as f:
            pickle.dump(graphs, f)

        if output_npy:
            np.save(output_npy, np.array([g["node_features"] for g in graphs], dtype=object))

        print(f"[OK] Parsed {len(graphs)} traces → {output_pkl}")
        return graphs


def main():
    converter = TraceToGraphConverter()
    cfg = TRACE_PARSER_CONFIG

    if not os.path.exists(cfg["input_file"]):
        print(f"[WARN] Missing input: {cfg['input_file']}")
        return

    converter.convert_traces_to_graphs(cfg["input_file"], cfg["output_pkl"], cfg["output_npy"])


if __name__ == "__main__":
    main()
