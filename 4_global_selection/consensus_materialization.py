"""
Build a consensus materialization for a combined refactoring by voting across
individual group ILP materializations.

Instead of re-running the ILP on the combined graph (which suffers from cost
model extrapolation on unfamiliar structures), this script leverages the
individual group ILPs that were already validated against actual BigQuery runs.

For each model:
  - If ≥threshold of individual groups agree on table/view, use that value.
  - Otherwise, keep the combined variant's original materialization.

Usage:
    python consensus_materialization.py \
        selected_0329_0400_7_groups_compile_0407_2242 \
        --individual-materializations \
            .../refactor_0329_0400_group_11_..._new_materialization.json \
            .../refactor_0329_0400_group_15_..._new_materialization.json \
            ...
"""

import os
import argparse
import json
from collections import Counter


BASE_DIR = "useful_files/tuva_analysis/for_materialization_tuning"
MATERIALIZATION_DIR = f"{BASE_DIR}/new_materializations"
HISTORY_DIR = "useful_files/tuva_dbt_run_official_history"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build consensus materialization from individual group ILP results."
    )
    parser.add_argument(
        "base_filename",
        help="Base filename for the combined variant "
             "(used to load original materialization from its manifest and name the output)",
    )
    parser.add_argument(
        "--individual-materializations",
        nargs="+",
        required=True,
        help="Paths to individual group ILP materialization JSON files",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.8,
        help="Fraction of individual groups that must agree to override "
             "the original materialization (default: 0.8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the consensus materialization JSON "
             "(default: {MATERIALIZATION_DIR}/{base_filename}_new_materialization.json)",
    )
    return parser.parse_args()


def load_original_materialization(base_filename):
    """Load the combined variant's original materialization directly from its manifest.json.

    Applies the same node filters as extract_features_for_refactored.py so the
    keyset matches what the individual-group ILPs voted over.
    """
    manifest_path = f"{HISTORY_DIR}/{base_filename}/manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    materialization = {}
    for uid, node in manifest["nodes"].items():
        mat = node.get("config", {}).get("materialized", "").lower()
        if mat in ["test", "incremental", "seed"]:
            continue
        if uid.startswith("operation."):
            continue
        if "model.elementary." in uid or uid == "model.the_tuva_project.data_quality__testing_summary":
            continue
        materialization[uid] = mat
    return materialization


def build_consensus(original_mat, individual_mats, threshold):
    """
    Build consensus materialization.

    For each model in the combined variant:
      1. Collect votes from individual groups that contain this model.
      2. If ≥threshold agree on a value, use it.
      3. Otherwise, keep the original materialization.

    Returns:
      consensus_mat: dict of model -> materialization
      report: dict with stats about the consensus process
    """
    consensus_mat = original_mat.copy()
    n_groups = len(individual_mats)
    min_votes = max(1, int(n_groups * threshold))

    stats = {
        "locked_to_table": [],
        "locked_to_view": [],
        "kept_original": [],
        "not_in_any_group": [],
    }

    for model, orig_val in original_mat.items():
        # Collect votes from individual groups
        votes = []
        for mat in individual_mats:
            if model in mat:
                votes.append(mat[model])

        if not votes:
            stats["not_in_any_group"].append(model)
            continue

        counter = Counter(votes)
        most_common_val, most_common_count = counter.most_common(1)[0]

        if most_common_count >= min_votes:
            consensus_mat[model] = most_common_val
            if most_common_val == "table" and orig_val != "table":
                stats["locked_to_table"].append(model)
            elif most_common_val != "table" and orig_val == "table":
                stats["locked_to_view"].append(model)
        else:
            stats["kept_original"].append(model)

    return consensus_mat, stats


def main():
    args = parse_args()

    # Load original materialization
    print(f"Loading original materialization for: {args.base_filename}")
    original_mat = load_original_materialization(args.base_filename)
    n_table_orig = sum(1 for v in original_mat.values() if v == "table")
    print(f"  Original: {n_table_orig} tables, {len(original_mat) - n_table_orig} views/other, {len(original_mat)} total")

    # Load individual group materializations
    individual_mats = []
    for path in args.individual_materializations:
        if not os.path.exists(path):
            print(f"  WARNING: Missing {path}, skipping")
            continue
        with open(path) as f:
            individual_mats.append(json.load(f))
        print(f"  Loaded: {os.path.basename(path)}")

    if not individual_mats:
        print("ERROR: No individual materializations loaded.")
        return

    print(f"\nBuilding consensus from {len(individual_mats)} groups "
          f"(threshold: {args.consensus_threshold:.0%})")

    # Build consensus
    consensus_mat, stats = build_consensus(
        original_mat, individual_mats, args.consensus_threshold,
    )

    # Report
    n_table = sum(1 for v in consensus_mat.values() if v == "table")
    n_view = sum(1 for v in consensus_mat.values() if v != "table")
    print(f"\nConsensus: {n_table} tables, {n_view} views, {len(consensus_mat)} total")
    print(f"  Locked to table by consensus: {len(stats['locked_to_table'])}")
    print(f"  Locked to view by consensus:  {len(stats['locked_to_view'])}")
    print(f"  Kept original (no consensus): {len(stats['kept_original'])}")
    print(f"  Not in any individual group:  {len(stats['not_in_any_group'])}")

    if stats["locked_to_view"]:
        print(f"\nModels locked to VIEW by consensus:")
        for m in sorted(stats["locked_to_view"]):
            print(f"  {m.split('.')[-1]}")

    if stats["locked_to_table"]:
        print(f"\nModels locked to TABLE by consensus:")
        for m in sorted(stats["locked_to_table"]):
            print(f"  {m.split('.')[-1]}")

    # Write output
    output_path = args.output
    if output_path is None:
        output_path = f"{MATERIALIZATION_DIR}/{args.base_filename}_new_materialization.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(consensus_mat, f, indent=4)
    print(f"\nConsensus materialization written to {output_path}")


if __name__ == "__main__":
    main()
