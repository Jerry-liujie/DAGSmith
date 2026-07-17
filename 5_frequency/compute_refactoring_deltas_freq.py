"""
Frequency-weighted cost delta estimation between original and refactored candidates.

Same as compute_refactoring_deltas.py, but applies frequency weights to B and U
before evaluating the objective, so deltas reflect real-world scheduling costs.

Usage:
    python compute_refactoring_deltas_freq.py \
        --prefix refactor_0329_0400_group \
        --frequency-profile freq_profiles/scenario_0.json
"""

import os
import re
import glob
import argparse
import json
from networkx.readwrite import json_graph

from utility.analyze import compute_dag_cost, evaluate_objective
from generate_frequency_profiles import propagate_to_cost_multipliers

BASE_DIR = "useful_files/tuva_analysis/for_materialization_tuning"
FEATURES_DIR = f"{BASE_DIR}/extracted_features_for_training"
GRAPH_DIR = f"{BASE_DIR}/graph_files"
MATERIALIZATION_DIR = f"{BASE_DIR}/new_materializations"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute frequency-weighted cost deltas between original and refactored candidates."
    )
    parser.add_argument(
        "--original",
        default="original_0415_0333",
        help="Base filename for the original variant",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix for auto-discovering candidates from new_materializations folder "
             "(e.g. 'refactor_0329_0400_group')",
    )
    parser.add_argument(
        "--frequency-profile",
        required=True,
        help="Path to a frequency profile JSON (from generate_frequency_profiles.py)",
    )
    parser.add_argument(
        "--compile-ts",
        default=None,
        help="Compile timestamp to filter candidates (e.g. '0512_1328'). "
             "Only candidates containing '_compile_{TS}' are kept. "
             "If omitted, all prefix matches are returned.",
    )
    parser.add_argument(
        "--refactor-dir",
        default=None,
        help="Path to the refactoring output dir (e.g. logs/refactored_slot_aware/0426_1422). "
             "When set, only groups with a group_*_final.json in this dir are discovered.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="List of candidate base filenames (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--original-materialization",
        default=None,
        help="Path to the original's materialization JSON file "
             "(if omitted, extracted from the original's features JSON)",
    )
    parser.add_argument(
        "--trained-models-bundle",
        default=f"{BASE_DIR}/trained_models/original_0415_0333_trained_models_bundle.joblib",
        help="Path to the trained models bundle (joblib)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the deltas JSON (auto-generated if omitted)",
    )
    args = parser.parse_args()
    if args.output is None:
        # Load scenario name for output path
        with open(args.frequency_profile) as f:
            scenario_name = json.load(f)["metadata"]["scenario_name"]
        args.output = (
            f"{BASE_DIR}/refactoring_benefit_estimation/"
            f"{args.prefix}_freq_{scenario_name}_refactoring_deltas.json"
        )
    return args


def artifact_paths(base_filename):
    return {
        "features": f"{FEATURES_DIR}/{base_filename}_extracted_features_for_training.json",
        "graph": f"{GRAPH_DIR}/{base_filename}_graph.json",
        "materialization": f"{MATERIALIZATION_DIR}/{base_filename}_new_materialization.json",
    }


def artifact_paths_freq(base_filename, scenario_name):
    """Paths for frequency-weighted materialization files."""
    return {
        "features": f"{FEATURES_DIR}/{base_filename}_extracted_features_for_training.json",
        "graph": f"{GRAPH_DIR}/{base_filename}_graph.json",
        "materialization": f"{MATERIALIZATION_DIR}/{base_filename}_freq_{scenario_name}_new_materialization.json",
    }


def has_all_artifacts(paths):
    for p in paths.values():
        if not os.path.exists(p):
            print(f"  Missing artifact: {p}")
            return False
    return True


def _valid_groups_from_refactor_dir(refactor_dir):
    """Return set of group IDs (e.g. {'group_5', 'group_12'}) present in refactor_dir."""
    valid = set()
    for f in glob.glob(os.path.join(refactor_dir, "group_*_final.json")):
        gid = os.path.basename(f).removesuffix("_final.json")
        valid.add(gid)
    return valid


def discover_candidates(prefix, scenario_name, compile_ts=None, refactor_dir=None):
    """Auto-discover candidates matching prefix from the new_materializations folder."""
    valid_groups = _valid_groups_from_refactor_dir(refactor_dir) if refactor_dir else None
    candidates = []
    suffix = f"_freq_{scenario_name}_new_materialization.json"
    for fname in sorted(os.listdir(MATERIALIZATION_DIR)):
        if fname.startswith(prefix) and fname.endswith(suffix):
            base = fname.removesuffix(suffix)
            if compile_ts and f"_compile_{compile_ts}" not in base:
                continue
            if valid_groups is not None:
                m = re.search(r'(group_\d+)', base)
                if m and m.group(1) not in valid_groups:
                    continue
            candidates.append(base)
    return candidates


def build_node_frequencies_from_graph(graph_path, root_freq, leaf_freq):
    """Re-propagate frequency through a candidate's DAG to cover new nodes."""
    with open(graph_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)

    # Build parent/child maps from the graph
    parent_map = {n: set() for n in G.nodes()}
    child_map = {n: set() for n in G.nodes()}
    for u, v in G.edges():
        child_map[u].add(v)
        parent_map[v].add(u)

    return propagate_to_cost_multipliers(root_freq, leaf_freq, parent_map, child_map)


def compute_dag_cost_freq(paths, trained_models_bundle, root_freq, leaf_freq):
    """
    Compute frequency-weighted DAG cost.

    1. Re-propagate frequencies through the candidate's DAG (covers new nodes)
    2. Call compute_dag_cost() to get unweighted B, U, t, edges
    3. Apply frequency weights: B[k] *= freq[k], U[(p,k)] *= freq[k]
    4. Re-evaluate with evaluate_objective()
    """
    node_frequencies = build_node_frequencies_from_graph(
        paths["graph"], root_freq, leaf_freq
    )

    result = compute_dag_cost(
        paths["features"], paths["graph"], paths["materialization"],
        trained_models_bundle,
    )

    B = result["B"]
    U = result["U"]
    t = result["t"]
    edges = result["edges"]

    # Apply frequency weights
    for k in list(B.keys()):
        B[k] *= node_frequencies.get(k, 1)

    for (p, k) in list(U.keys()):
        U[(p, k)] *= node_frequencies.get(k, 1)

    freq_cost = evaluate_objective(B, U, edges, t)

    return {
        "cost": freq_cost,
        "unweighted_cost": result["cost"],
        "B": B,
        "U": U,
        "t": t,
        "num_models": result["num_models"],
        "edges": edges,
    }


def main():
    args = parse_args()
    original = args.original

    # Load frequency profile
    with open(args.frequency_profile) as f:
        freq_profile = json.load(f)
    root_freq = freq_profile["root_frequencies"]
    leaf_freq = freq_profile["leaf_frequencies"]
    scenario_name = freq_profile["metadata"]["scenario_name"]

    print(f"Frequency profile: {args.frequency_profile}")
    print(f"Scenario: {scenario_name}")

    candidates = args.candidates
    if candidates is None:
        candidates = discover_candidates(args.prefix, scenario_name, compile_ts=args.compile_ts,
                                         refactor_dir=args.refactor_dir)
        print(f"Auto-discovered {len(candidates)} candidates: {candidates}")

    # compute original cost (use features-extracted materialization, matching regular pipeline)
    orig_paths = artifact_paths(original)
    for label, p in [("features", orig_paths["features"]), ("graph", orig_paths["graph"])]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {label} artifact for original: {p}")
            return

    orig_mat_path = args.original_materialization
    if orig_mat_path is None:
        import tempfile
        with open(orig_paths["features"]) as f:
            mat = json.load(f).get("materialization", {})
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_materialization.json", delete=False,
        )
        json.dump(mat, tmp, indent=2)
        tmp.close()
        orig_mat_path = tmp.name
        print(f"Extracted original materialization from features JSON -> {orig_mat_path}")
    elif not os.path.exists(orig_mat_path):
        print(f"ERROR: Original materialization file not found: {orig_mat_path}")
        return

    orig_paths["materialization"] = orig_mat_path

    print(f"Computing frequency-weighted cost for original: {original}")
    original_result = compute_dag_cost_freq(
        orig_paths, args.trained_models_bundle, root_freq, leaf_freq,
    )
    original_cost = original_result["cost"]
    print(f"  Original cost (freq-weighted): {original_cost:,.2f}  "
          f"(unweighted: {original_result['unweighted_cost']:,.2f})  "
          f"({original_result['num_models']} models)")

    # compute candidate costs
    candidate_results = {}
    for cand in candidates:
        cand_paths = artifact_paths_freq(cand, scenario_name)
        if not has_all_artifacts(cand_paths):
            print(f"  SKIP {cand}: missing artifacts")
            continue

        print(f"Computing frequency-weighted cost for candidate: {cand}")
        result = compute_dag_cost_freq(
            cand_paths, args.trained_models_bundle, root_freq, leaf_freq,
        )
        delta = original_cost - result["cost"]
        candidate_results[cand] = {
            "cost": result["cost"],
            "unweighted_cost": result["unweighted_cost"],
            "delta": delta,
            "num_models": result["num_models"],
        }
        print(f"  Cost: {result['cost']:,.2f}  delta: {delta:,.2f}  ({result['num_models']} models)")

    # write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "original": original,
        "original_cost": original_cost,
        "original_unweighted_cost": original_result["unweighted_cost"],
        "frequency_scenario": scenario_name,
        "frequency_profile": args.frequency_profile,
        "candidates": candidate_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDeltas written to {args.output}")


if __name__ == "__main__":
    main()
