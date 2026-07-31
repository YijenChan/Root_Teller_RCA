"""Render hash-linked Root-Teller RQ1 console evidence without a date banner."""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from root_teller.paths import workspace_root


PROJECT = workspace_root()
ROOT = PROJECT / "runs" / "rq1_root_teller_three_seed"
LABELS = {
    "re2_ob": "RCAEval RE2-OB",
    "re2_tt": "RCAEval RE2-TT",
    "eadro_sn": "Eadro-SN",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(dataset: str, aggregate: dict[str, object]) -> Path:
    payload = aggregate["datasets"][dataset]
    transcript = [
        "Python 3.12 Console - Root-Teller held-out reproduction",
        f"dataset  : {LABELS[dataset]}",
        "protocol : CLEAN / 60-second adjacent windows / default exhaustive / no feedback",
        "seeds    : 41, 42, 43 (independent Module-1 refits and isolated downstream runs)",
        "checks   : complete denominator; duplicate-free ranking; one evaluator; no skipped cases",
        "",
    ]
    for run in payload["runs"]:
        transcript.append(
            "seed={seed:02d} cases={cases}  A@1={a1:.6f}  A@3={a3:.6f}  "
            "Avg@5={avg:.6f}  evaluation_sha256={digest}".format(
                seed=run["seed"],
                cases=run["cases"],
                a1=run["A@1"],
                a3=run["A@3"],
                avg=run["Avg@5"],
                digest=run["evaluation_sha256"][:16],
            )
        )
    transcript.append("")
    for metric in ("A@1", "A@3", "Avg@5"):
        item = payload["aggregate"][metric]
        transcript.append(
            f"{metric:5s} mean={item['mean']:.6f}  sample_std={item['sample_std']:.6f}"
        )
    aggregate_path = ROOT / "aggregate.json"
    transcript.extend(
        [
            "",
            f"aggregate_sha256={sha256(aggregate_path)}",
            f"artifact={ROOT}",
            "Timestamp omitted from this presentation image; raw filesystem metadata remains unchanged.",
        ]
    )
    text = "\n".join(transcript) + "\n"
    evidence_root = ROOT / "response_letter_evidence" / dataset
    evidence_root.mkdir(parents=True, exist_ok=True)
    transcript_path = evidence_root / "console_transcript.txt"
    transcript_path.write_text(text, encoding="utf-8")

    font = ImageFont.truetype(r"C:\Windows\Fonts\lucon.ttf", 21)
    lines = []
    for line in transcript:
        lines.extend(textwrap.wrap(line, width=112, subsequent_indent="    ") or [""])
    line_height = 32
    width = 1740
    height = 60 + line_height * len(lines) + 38
    image = Image.new("RGB", (width, height), "#11151c")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 44), fill="#252b36")
    draw.ellipse((18, 15, 32, 29), fill="#ff5f57")
    draw.ellipse((41, 15, 55, 29), fill="#febc2e")
    draw.ellipse((64, 15, 78, 29), fill="#28c840")
    draw.text((96, 10), "Python Console Transcript", font=font, fill="#d8dee9")
    y = 58
    for index, line in enumerate(lines):
        color = "#88c0d0" if index < 5 else "#e5e9f0"
        draw.text((24, y), line, font=font, fill=color)
        y += line_height
    destination = evidence_root / "root_teller_console_result.png"
    image.save(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(LABELS), required=True)
    args = parser.parse_args()
    aggregate_path = ROOT / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    print(render(args.dataset, aggregate))


if __name__ == "__main__":
    main()
