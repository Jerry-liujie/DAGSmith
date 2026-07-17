import os
import json
import argparse
from datetime import datetime
from utility.analyze import load_manifest, build_parent_child_maps


def main():
    parser = argparse.ArgumentParser(
        description="Extract a single subgraph containing ALL models from a dbt manifest."
    )
    parser.add_argument("--manifest-path", required=True, help="Path to dbt manifest.json")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: logs/extracted_subgraphs/TIMESTAMP/)")
    parser.add_argument("--group-id", default="group_0", help="Group ID for the output file (default: group_0)")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"logs/extracted_subgraphs/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_manifest(args.manifest_path)
    parent_map, child_map = build_parent_child_maps(manifest)
    nodes = manifest["nodes"]

    model_uids = sorted(uid for uid in nodes if uid.startswith("model."))
    model_set = set(model_uids)

    # edges: parent -> [children], filtered to model-to-model within the set
    edges = {}
    for uid in model_uids:
        children = sorted(model_set & set(child_map.get(uid, [])))
        if children:
            edges[uid] = children

    # roots: models with no in-set parents
    roots = []
    for uid in model_uids:
        parents_in_set = model_set & set(parent_map.get(uid, []))
        if not parents_in_set:
            roots.append(uid)

    # has_external_children: models consumed by models NOT in the set.
    # Since we include all models, this is empty.
    has_external_children = {}
    for uid in model_uids:
        external_model_children = {
            c for c in child_map.get(uid, [])
            if c.startswith("model.") and c not in model_set
        }
        if external_model_children:
            has_external_children[uid] = True

    # collect model data
    models_data = {}
    for uid in model_uids:
        node = nodes[uid]
        cfg = node.get("config", {})
        models_data[uid] = {
            "name": node.get("name", uid),
            "raw_code": node.get("raw_code"),
            "materialized": cfg.get("materialized"),
        }

    output = {
        "group_id": args.group_id,
        "group_metadata": {
            "score": None,
            "mass": None,
            "num_models": len(model_uids),
        },
        "extraction_params": {"mode": "all_models"},
        "subgraph": {
            "all_nodes": model_uids,
            "component_nodes": model_uids,
            "context_nodes": [],
            "roots": sorted(roots),
            "edges": edges,
            "has_external_children": has_external_children,
        },
        "models": models_data,
    }

    output_path = os.path.join(output_dir, f"{args.group_id}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Models:    {len(model_uids)}")
    print(f"Roots:     {len(roots)}")
    print(f"Edges:     {len(edges)}")
    print(f"Immutable: {len(has_external_children)}")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
