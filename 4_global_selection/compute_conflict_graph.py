import os
import re
import json
import glob
import argparse
from itertools import combinations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute conflict graph between LLM refactoring groups based on overlapping changed models."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing group_*_final.json files (e.g. logs/refactored/0329_0400)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for conflict graph JSON (default: {input_dir}/conflict_graph.json)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.input_dir, "conflict_graph.json")
    return args


def extract_changed_uids(filepath):
    """Extract UIDs of models that were rewritten, newly created, or removed."""
    with open(filepath) as f:
        data = json.load(f)
    changed = set()
    for model in data.get("models", []):
        if model.get("rewritten") or model.get("is_new") or model.get("removed"):
            changed.add(model["uid"])
    return changed


def main():
    args = parse_args()

    # discover group_*_final.json files
    pattern = os.path.join(args.input_dir, "group_*_final.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No group_*_final.json files found in {args.input_dir}")
        return

    # extract changed UIDs per group
    groups = {}
    for filepath in files:
        basename = os.path.basename(filepath)
        m = re.match(r"(group_\d+)_final\.json", basename)
        if not m:
            continue
        group_id = m.group(1)
        groups[group_id] = extract_changed_uids(filepath)

    print(f"Found {len(groups)} groups in {args.input_dir}")
    for gid in sorted(groups, key=lambda g: int(g.split("_")[1])):
        print(f"  {gid}: {len(groups[gid])} changed model(s)")

    # compute pairwise conflicts
    conflict_pairs = []
    for g1, g2 in combinations(sorted(groups), 2):
        overlap = groups[g1] & groups[g2]
        if overlap:
            conflict_pairs.append({
                "groups": [g1, g2],
                "overlapping_uids": sorted(overlap),
            })

    print(f"\nConflict pairs found: {len(conflict_pairs)}")
    for cp in conflict_pairs:
        print(f"  {cp['groups'][0]} <-> {cp['groups'][1]}: {len(cp['overlapping_uids'])} overlapping uid(s)")

    # write output
    output = {
        "source_directory": args.input_dir,
        "groups": {gid: sorted(uids) for gid, uids in sorted(groups.items())},
        "conflict_pairs": conflict_pairs,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nConflict graph written to {args.output}")


if __name__ == "__main__":
    main()
