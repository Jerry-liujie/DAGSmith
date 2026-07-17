import json
import re
import argparse
from collections import defaultdict

import pyomo.environ as pyo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select optimal non-conflicting combination of refactoring candidates."
    )
    parser.add_argument(
        "deltas_json",
        help="Path to the refactoring deltas JSON file",
    )
    parser.add_argument(
        "--conflicts",
        default=None,
        help="Path to conflict graph JSON (from compute_conflict_graph.py). "
             "If omitted, reads 'conflict_pairs' key from deltas JSON, or runs unconstrained.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for selection JSON (default: <deltas>_selection.json)",
    )
    return parser.parse_args()


def parse_group_id(candidate_name):
    """Extract group ID from a candidate name.

    Handles both legacy 'cc_5_0203_1207' and new 'refactor_0329_0400_group_0_compile_...' formats.
    Returns normalized form like 'group_0'.
    """
    m = re.search(r"(?:cc|group)_(\d+)", candidate_name)
    if m:
        return f"group_{m.group(1)}"
    return None


def load_conflict_pairs(args, data):
    """Load conflict pairs from --conflicts file, from deltas JSON, or return empty list."""
    if args.conflicts:
        with open(args.conflicts) as f:
            conflict_data = json.load(f)
        return [
            {p["groups"][0], p["groups"][1]}
            for p in conflict_data["conflict_pairs"]
        ]
    if "conflict_pairs" in data:
        return [
            {p["groups"][0], p["groups"][1]}
            for p in data["conflict_pairs"]
        ]
    print("WARNING: No conflict information provided. Running without conflict constraints.")
    return []


def main():
    args = parse_args()

    with open(args.deltas_json) as f:
        data = json.load(f)

    original_cost = data["original_cost"]
    candidates = data["candidates"]  # {name: {cost, delta, num_models}}

    conflict_pairs = load_conflict_pairs(args, data)

    # group variants by group id
    groups = defaultdict(list)  # group_id -> [candidate_names]
    for cand in candidates:
        g = parse_group_id(cand)
        if g is None:
            print(f"WARNING: cannot parse group id from '{cand}', skipping")
            continue
        groups[g].append(cand)

    all_cands = [c for g in groups.values() for c in g]
    all_groups = sorted(groups.keys())

    print(f"Candidates: {len(all_cands)} variants across {len(all_groups)} groups")
    for g in all_groups:
        variants = groups[g]
        best = max(variants, key=lambda c: candidates[c]["delta"])
        print(f"  {g}: {len(variants)} variant(s), best delta = {candidates[best]['delta']:,.2f} ({best})")

    if conflict_pairs:
        print(f"\nConflict constraints: {len(conflict_pairs)}")
        for pair in conflict_pairs:
            a, b = sorted(pair)
            print(f"  {a} <-> {b}")

    # --- build ILP ---
    model = pyo.ConcreteModel()

    model.cands = pyo.Set(initialize=all_cands)
    model.groups = pyo.Set(initialize=all_groups)

    # binary: select this candidate?
    model.x = pyo.Var(model.cands, within=pyo.Binary)
    # auxiliary: is any variant from this group selected?
    model.z = pyo.Var(model.groups, within=pyo.Binary)

    # objective: maximize total delta
    model.obj = pyo.Objective(
        expr=sum(candidates[c]["delta"] * model.x[c] for c in all_cands),
        sense=pyo.maximize,
    )

    # at most one variant per group, linked to z
    model.group_cons = pyo.ConstraintList()
    for g in all_groups:
        variants = groups[g]
        model.group_cons.add(sum(model.x[c] for c in variants) <= model.z[g])

    # conflict constraints between groups
    model.conflict_cons = pyo.ConstraintList()
    for pair in conflict_pairs:
        a, b = list(pair)
        if a in all_groups and b in all_groups:
            model.conflict_cons.add(model.z[a] + model.z[b] <= 1)

    # solve
    solver = pyo.SolverFactory("highs")
    if solver is None or not solver.available():
        solver = pyo.SolverFactory("cbc")
    res = solver.solve(model, tee=False)

    # extract solution
    selected = []
    for c in all_cands:
        if round(pyo.value(model.x[c])) == 1:
            selected.append({
                "candidate": c,
                "group": parse_group_id(c),
                "delta": candidates[c]["delta"],
                "cost": candidates[c]["cost"],
            })

    selected.sort(key=lambda x: x["group"])
    total_benefit = sum(s["delta"] for s in selected)
    selected_groups = {s["group"] for s in selected}

    # groups excluded by conflict
    excluded = set()
    for pair in conflict_pairs:
        overlap = pair & selected_groups
        if overlap:
            excluded |= (pair - overlap)
    excluded -= selected_groups

    not_selected = sorted(set(all_cands) - {s["candidate"] for s in selected})

    # output
    print(f"\n{'='*60}")
    print(f"OPTIMAL COMBINATION")
    print(f"{'='*60}")
    print(f"Original cost:        {original_cost:>14,.2f}")
    print(f"Total expected benefit: {total_benefit:>13,.2f}")
    print(f"Optimized cost:       {original_cost - total_benefit:>14,.2f}")
    print(f"\nSelected refactorings ({len(selected)}):")
    for s in selected:
        print(f"  {s['candidate']:40s}  delta = {s['delta']:>12,.2f}")
    if excluded:
        print(f"\nExcluded by conflict: {sorted(excluded)}")
    if not_selected:
        print(f"Not selected: {not_selected}")

    # write JSON
    output_path = args.output or args.deltas_json.replace(".json", "_selection.json")
    output = {
        "original_cost": original_cost,
        "total_expected_benefit": total_benefit,
        "optimized_cost": original_cost - total_benefit,
        "selected": selected,
        "excluded_by_conflict": sorted(excluded),
        "not_selected": not_selected,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSelection written to {output_path}")


if __name__ == "__main__":
    main()
