"""
Build a consensus materialization for a combined refactoring by voting across
individual group frequency-weighted ILP materializations.

Same as consensus_materialization.py, but reads from frequency-weighted
materialization files (*_freq_{scenario}_new_materialization.json).

Usage:
    python consensus_materialization_freq.py \
        selected_0329_0400_7_groups_compile_0407_2242 \
        --frequency-profile freq_profiles/scenario_0.json \
        --individual-base-filenames \
            refactor_0329_0400_group_11_compile_0330_0438 \
            refactor_0329_0400_group_15_compile_0330_0438 \
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
        description="Build consensus materialization from individual group "
                    "frequency-weighted ILP results."
    )
    parser.add_argument(
        "base_filename",
        help="Base filename for the combined variant",
    )
    parser.add_argument(
        "--frequency-profile",
        required=True,
        help="Path to a frequency profile JSON (for scenario name resolution)",
    )
    parser.add_argument(
        "--individual-base-filenames",
        nargs="+",
        required=True,
        help="Base filenames of individual group variants "
             "(frequency materialization files are auto-resolved using the scenario name)",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.8,
        help="Fraction of individual groups that must agree (default: 0.8)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: auto-generated with scenario name)",
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

    # Load frequency profile for scenario name
    with open(args.frequency_profile) as f:
        freq_profile = json.load(f)
    scenario_name = freq_profile["metadata"]["scenario_name"]
    print(f"Frequency scenario: {scenario_name}")

    # Load original materialization
    print(f"Loading original materialization for: {args.base_filename}")
    original_mat = load_original_materialization(args.base_filename)
    n_table_orig = sum(1 for v in original_mat.values() if v == "table")
    print(f"  Original: {n_table_orig} tables, {len(original_mat) - n_table_orig} views/other")

    # Load individual group frequency-weighted materializations
    individual_mats = []
    for base in args.individual_base_filenames:
        path = (f"{MATERIALIZATION_DIR}/{base}"
                f"_freq_{scenario_name}_new_materialization.json")
        if not os.path.exists(path):
            # Fall back to non-freq materialization if freq version doesn't exist
            fallback = f"{MATERIALIZATION_DIR}/{base}_new_materialization.json"
            if os.path.exists(fallback):
                print(f"  WARNING: No freq materialization for {base}, using non-freq fallback")
                path = fallback
            else:
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
        output_path = (
            f"{MATERIALIZATION_DIR}/{args.base_filename}"
            f"_freq_{scenario_name}_new_materialization.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(consensus_mat, f, indent=4)
    print(f"\nConsensus materialization written to {output_path}")


if __name__ == "__main__":
    main()
