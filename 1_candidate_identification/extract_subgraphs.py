import os
import json
import argparse
from datetime import datetime
from utility.analyze import load_manifest, build_parent_child_maps
from utility.subgraph import SubgraphAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Extract refactoring subgraphs from model groups.")
    parser.add_argument("--groups-file", required=True, help="Path to groups JSON (e.g., 0324_dataflow_analysis_top_groups.json)")
    parser.add_argument("--manifest-path", required=True, help="Path to dbt manifest.json")
    parser.add_argument("--upstream-depth", type=int, default=1, help="BFS hops upstream from component nodes (default: 1)")
    parser.add_argument("--downstream-depth", type=int, default=1, help="BFS hops downstream from component nodes (default: 1)")
    parser.add_argument("--include-siblings", action="store_true", default=True, help="Include sibling models (default: True)")
    parser.add_argument("--no-siblings", action="store_true", help="Disable sibling inclusion")
    parser.add_argument("--max-nodes", type=int, default=60, help="Max subgraph size; groups exceeding this are skipped (default: 60)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: logs/extracted_subgraphs/TIMESTAMP/)")
    parser.add_argument("--groups", nargs="*", default=None, help="Specific group IDs to process (default: all)")
    args = parser.parse_args()

    include_siblings = args.include_siblings and not args.no_siblings

    # output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"logs/extracted_subgraphs/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    # load manifest and build DAG
    manifest = load_manifest(args.manifest_path)
    parent_map, child_map = build_parent_child_maps(manifest)
    nodes = manifest["nodes"]

    analyzer = SubgraphAnalyzer(parents=parent_map, children=child_map)

    # load groups
    with open(args.groups_file) as f:
        groups = json.load(f)

    selected = args.groups if args.groups else sorted(groups.keys())

    extraction_params = {
        "upstream_depth": args.upstream_depth,
        "downstream_depth": args.downstream_depth,
        "include_siblings": include_siblings,
        "max_nodes": args.max_nodes,
    }

    # summary table header
    print(f"{'Group':<16} | {'Components':>10} | {'Context':>7} | {'Total':>5} | {'Score':>6}")
    print("-" * 60)

    for group_id in selected:
        if group_id not in groups:
            print(f"WARNING: {group_id} not found in groups file, skipping")
            continue

        group_info = groups[group_id]
        component_uids = group_info["models"]

        sg = analyzer.extract_neighborhood_subgraph(
            component_uids,
            upstream_depth=args.upstream_depth,
            downstream_depth=args.downstream_depth,
            include_siblings=include_siblings,
            max_nodes=args.max_nodes,
        )

        if sg is None:
            print(f"{group_id:<16} | SKIPPED (exceeds {args.max_nodes} nodes)")
            continue

        # Filter out seed nodes — small pre-seeded reference tables
        seed_uids = {n for n in sg.nodes if n.startswith("seed.")}
        filtered_nodes = sg.nodes - seed_uids

        context_nodes = sorted(filtered_nodes - sg.component_nodes)
        component_list = sorted(sg.component_nodes - seed_uids)
        all_nodes_list = sorted(filtered_nodes)

        # collect model data
        models_data = {}
        for uid in all_nodes_list:
            node = nodes.get(uid)
            if node is None:
                # might be a source/seed not in nodes dict
                models_data[uid] = {
                    "name": uid.split(".")[-1] if "." in uid else uid,
                    "raw_code": None,
                    "materialized": None,
                }
                continue

            cfg = node.get("config", {}) if isinstance(node, dict) else {}
            models_data[uid] = {
                "name": node.get("name", uid),
                "raw_code": node.get("raw_code"),
                "materialized": cfg.get("materialized"),
            }

        output = {
            "group_id": group_id,
            "group_metadata": {
                "score": group_info.get("score"),
                "mass": group_info.get("mass"),
                "num_models": group_info.get("num_models"),
            },
            "extraction_params": extraction_params,
            "subgraph": {
                "all_nodes": all_nodes_list,
                "component_nodes": component_list,
                "context_nodes": context_nodes,
                "roots": sorted(sg.roots - seed_uids),
                "edges": {k: sorted(v - seed_uids) for k, v in sg.edges.items() if k not in seed_uids and (v - seed_uids)},
                "has_external_children": {k: v for k, v in sg.has_external_children.items() if v and k not in seed_uids},
            },
            "models": models_data,
        }

        output_path = os.path.join(output_dir, f"{group_id}.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        score = group_info.get("score", "?")
        print(f"{group_id:<16} | {len(component_list):>10} | {len(context_nodes):>7} | {len(all_nodes_list):>5} | {score:>6}")

    print(f"\nOutput written to {output_dir}")


if __name__ == "__main__":
    main()
