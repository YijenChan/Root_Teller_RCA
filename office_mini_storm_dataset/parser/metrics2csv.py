# metrics2csv.py (anonymized & open-source safe version)
# Converts Fluent Bit JSON logs of metrics into CSV format.
# Automatically skips or normalizes sensitive hardware fields (e.g., Mem.total).

import json
import csv
import os
from config import METRICS_PARSER_CONFIG

# Sensitive or machine-specific metric keys to exclude or normalize
SENSITIVE_KEYS = {"Mem.total", "Swap.total", "cpu_num", "hostname"}

def sanitize_metrics(metrics: dict) -> dict:
    """Remove or normalize sensitive metrics before export."""
    clean = {}
    for k, v in metrics.items():
        if k in SENSITIVE_KEYS:
            # Normalize numeric values to relative scale (e.g., ratio)
            if isinstance(v, (int, float)) and v != 0:
                clean[k] = round(v / max(v, 1), 4)
            else:
                continue  # drop non-numeric sensitive fields
        else:
            clean[k] = v
    return clean

def main():
    in_file = METRICS_PARSER_CONFIG["input_file"]
    out_file = METRICS_PARSER_CONFIG["output_file"]

    if not os.path.exists(in_file):
        print(f"[WARN] Input file not found: {in_file}")
        return

    with open(in_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", newline="", encoding="utf-8") as fout:

        writer = None
        total_lines = 0
        skipped = 0

        for line in fin:
            try:
                obj = json.loads(line.strip())
                if "metrics" not in obj:
                    skipped += 1
                    continue

                metrics = sanitize_metrics(obj["metrics"])
                record = {"timestamp": obj.get("timestamp", 0), **metrics}

                if writer is None:
                    # Initialize CSV writer on first valid line
                    fieldnames = list(record.keys())
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()

                writer.writerow(record)
                total_lines += 1

            except json.JSONDecodeError:
                skipped += 1
                continue

    print(f"[OK] CSV generated: {out_file}")
    print(f"Processed lines: {total_lines}, Skipped: {skipped}")

if __name__ == "__main__":
    main()
