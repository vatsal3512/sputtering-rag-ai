"""
run_pipeline.py
───────────────
Single entry-point orchestrator for the full 6-stage sputtering RAG pipeline.
Replaces manually running 6 disconnected scripts.

Usage:
    python run_pipeline.py              # Run all stages
    python run_pipeline.py --stage 5    # Run only Stage 5
    python run_pipeline.py --from 3     # Run Stages 3 → 6
    python run_pipeline.py --list       # Print stage descriptions and exit
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from config_loader import config

# =============================================================================
# STAGE REGISTRY
# =============================================================================
STAGES = [
    {
        "id":     1,
        "name":   "GROBID XML Parser",
        "script": "full_grobid_pipeline.py",
        "desc":   "Parse GROBID TEI XML → per-paper structured_data.json",
        "output_check": config.path("processed_articles_dir"),  # dir must be non-empty
    },
    {
        "id":     2,
        "name":   "Gemini NER Extractor",
        "script": "master_data_extraction.py",
        "desc":   "3-shot Gemini NER → *_extracted.json per paper (needs API keys in .env)",
        "output_check": config.path("extracted_data_dir"),
    },
    {
        "id":     3,
        "name":   "EDA & Unit Normalization",
        "script": "post_processing_eda.py",
        "desc":   "Normalize units to SI (Pa, nm, °C, W) → sputtering_database_clean.csv",
        "output_check": config.path("cleaned_csv"),
    },
    {
        "id":     4,
        "name":   "Sputtering Filter",
        "script": "post_processing2.py",
        "desc":   "Filter non-sputtering rows → sputtering_database_clean_final.csv",
        "output_check": config.path("final_csv"),
    },
    {
        "id":     5,
        "name":   "Vector DB Builder",
        "script": "build_vector_db.py",
        "desc":   "Embed CSV into ChromaDB with SciBERT (no API key needed)",
        "output_check": os.path.join(config.path("vector_database"), "chroma.sqlite3"),
    },
    {
        "id":     6,
        "name":   "Streamlit Dashboard",
        "script": "app.py",
        "desc":   "Launch the Streamlit UI (Ctrl+C to stop)",
        "output_check": None,  # interactive — never skip
        "is_server": True,
    },
]


# =============================================================================
# HELPERS
# =============================================================================
def _already_done(stage: dict) -> bool:
    """Return True if the stage output already exists (skip logic)."""
    if stage.get("is_server"):
        return False  # always run servers
    check = stage.get("output_check")
    if not check:
        return False
    p = Path(check)
    if p.is_dir():
        return any(p.iterdir())  # non-empty directory
    return p.exists()


def _run_stage(stage: dict, force: bool = False) -> bool:
    """Run a single stage. Returns True on success."""
    sid    = stage["id"]
    name   = stage["name"]
    script = stage["script"]

    print(f"\n{'─'*60}")
    print(f"  Stage {sid}: {name}")
    print(f"  {stage['desc']}")
    print(f"{'─'*60}")

    if not force and _already_done(stage):
        print(f"  ⏭️  Output already exists — skipping.")
        print(f"     (Use --force to override)")
        return True

    script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"  ❌ Script not found: {script_path}")
        return False

    # Use the same Python interpreter that's running this script
    cmd = [sys.executable, str(script_path)]

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ Stage {sid} complete — {elapsed:.1f}s")
        return True
    else:
        print(f"\n  ❌ Stage {sid} FAILED (exit code {result.returncode})")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Sputtering RAG AI Pipeline Orchestrator"
    )
    parser.add_argument(
        "--stage", type=int, metavar="N",
        help="Run only stage N (1–6)"
    )
    parser.add_argument(
        "--from", dest="from_stage", type=int, metavar="N",
        help="Run stages N through 6"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if output already exists"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print stage descriptions and exit"
    )
    args = parser.parse_args()

    # ── --list mode ──────────────────────────────────────────────────────────
    if args.list:
        print("\n  Sputtering RAG AI — Pipeline Stages\n")
        for s in STAGES:
            done = "✅" if _already_done(s) else "⬜"
            print(f"  {done} Stage {s['id']}: {s['name']}")
            print(f"       {s['desc']}\n")
        return

    # ── Select stages to run ─────────────────────────────────────────────────
    if args.stage:
        stages_to_run = [s for s in STAGES if s["id"] == args.stage]
        if not stages_to_run:
            print(f"❌ Invalid stage: {args.stage}. Must be 1–{len(STAGES)}.")
            sys.exit(1)
    elif args.from_stage:
        stages_to_run = [s for s in STAGES if s["id"] >= args.from_stage]
    else:
        stages_to_run = STAGES

    # ── Banner ───────────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("   Sputtering RAG AI — Pipeline Orchestrator")
    print("═"*60)
    print(f"   Running {len(stages_to_run)} stage(s): "
          f"{[s['id'] for s in stages_to_run]}")
    print(f"   Force re-run: {args.force}")

    # ── Execute ───────────────────────────────────────────────────────────────
    total_start = time.time()
    results = {}

    for stage in stages_to_run:
        success = _run_stage(stage, force=args.force)
        results[stage["id"]] = success
        if not success and not stage.get("is_server"):
            print(f"\n🛑 Stopping pipeline at Stage {stage['id']} due to failure.")
            break

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  Pipeline Summary — {total_elapsed:.1f}s total")
    print(f"{'═'*60}")
    for sid, success in results.items():
        name = next(s["name"] for s in STAGES if s["id"] == sid)
        icon = "✅" if success else "❌"
        print(f"  {icon} Stage {sid}: {name}")
    print()


if __name__ == "__main__":
    main()
