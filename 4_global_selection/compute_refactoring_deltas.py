import os
import re
import glob
import argparse
import json

from utility.analyze import compute_dag_cost

BASE_DIR = "useful_files/tuva_analysis/for_materialization_tuning"
FEATURES_DIR = f"{BASE_DIR}/extracted_features_for_training"
GRAPH_DIR = f"{BASE_DIR}/graph_files"
MATERIALIZATION_DIR = f"{BASE_DIR}/new_materializations"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cost deltas between original and refactored candidates."
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
        help="Output path for the deltas JSON "
             "(default: {BASE_DIR}/refactoring_benefit_estimation/{prefix}_refactoring_deltas.json)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"{BASE_DIR}/refactoring_benefit_estimation/{args.prefix}_refactoring_deltas.json"
    return args


def artifact_paths(base_filename):
    return {
        "features": f"{FEATURES_DIR}/{base_filename}_extracted_features_for_training.json",
        "graph": f"{GRAPH_DIR}/{base_filename}_graph.json",
        "materialization": f"{MATERIALIZATION_DIR}/{base_filename}_new_materialization.json",
    }


def has_all_artifacts(base_filename):
    paths = artifact_paths(base_filename)
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


def discover_candidates(prefix, compile_ts=None, refactor_dir=None):
    """Auto-discover candidates matching prefix from the new_materializations folder."""
    valid_groups = _valid_groups_from_refactor_dir(refactor_dir) if refactor_dir else None
    candidates = []
    for fname in sorted(os.listdir(MATERIALIZATION_DIR)):
        if fname.startswith(prefix) and fname.endswith("_new_materialization.json"):
            base = fname.removesuffix("_new_materialization.json")
            if compile_ts and f"_compile_{compile_ts}" not in base:
                continue
            if valid_groups is not None:
                m = re.search(r'(group_\d+)', base)
                if m and m.group(1) not in valid_groups:
                    continue
            candidates.append(base)
    return candidates


def main():
    args = parse_args()
    original = args.original
    candidates = args.candidates

    if candidates is None:
        candidates = discover_candidates(args.prefix, compile_ts=args.compile_ts,
                                         refactor_dir=args.refactor_dir)
        print(f"Auto-discovered {len(candidates)} candidates: {candidates}")

    # compute original cost (original does not need a new_materialization file)
    orig_paths = artifact_paths(original)
    for label, p in [("features", orig_paths["features"]), ("graph", orig_paths["graph"])]:
        if not os.path.exists(p):
            print(f"ERROR: Missing {label} artifact for original: {p}")
            return

    # resolve original materialization path
    orig_mat_path = args.original_materialization
    if orig_mat_path is None:
        # extract materialization from the features JSON and write to a temp file
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

    print(f"Computing cost for original: {original}")
    original_result = compute_dag_cost(
        orig_paths["features"], orig_paths["graph"], orig_mat_path,
        args.trained_models_bundle,
    )
    original_cost = original_result["cost"]
    print(f"  Original cost: {original_cost:,.2f}  ({original_result['num_models']} models)")

    # compute candidate costs
    candidate_results = {}
    for cand in candidates:
        if not has_all_artifacts(cand):
            print(f"  SKIP {cand}: missing artifacts")
            continue

        cand_paths = artifact_paths(cand)
        print(f"Computing cost for candidate: {cand}")
        result = compute_dag_cost(
            cand_paths["features"], cand_paths["graph"], cand_paths["materialization"],
            args.trained_models_bundle,
        )
        delta = original_cost - result["cost"]
        candidate_results[cand] = {
            "cost": result["cost"],
            "delta": delta,
            "num_models": result["num_models"],
        }
        print(f"  Cost: {result['cost']:,.2f}  delta: {delta:,.2f}  ({result['num_models']} models)")

    # write output
    output = {
        "original": original,
        "original_cost": original_cost,
        "candidates": candidate_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDeltas written to {args.output}")


if __name__ == "__main__":
    main()
