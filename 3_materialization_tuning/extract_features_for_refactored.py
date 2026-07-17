"""
Extract features for a refactored dbt candidate WITHOUT requiring dbt run.

Uses:
  - Refactored manifest.json (from `dbt compile` on the refactored branch)
    for SQL features, DAG structure, and materialization config
  - Original run_results.json for table cardinalities of existing models
  - Trained models bundle to predict cardinality for new models
  - Baseline cardinality for source/seed tables

Produces the same output format as extract_features_for_training.py so that
downstream scripts (iterative_local_linearization.py, compute_refactoring_deltas.py)
work unchanged.
"""

import json
import argparse
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from copy import deepcopy
import numpy as np
import os
import joblib

from utility.analyze import (
    analyze_sql_features_for_slot_time,
    analyze_sql_features_for_cardinality,
    count_refs,
    add_scaled_inplace,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features for a refactored candidate using original cardinalities."
    )
    parser.add_argument(
        "base_filename",
        help="Base filename for the refactored candidate "
             "(resolves to useful_files/tuva_dbt_run_official_history/{base_filename}/)",
    )
    parser.add_argument(
        "--original-run-results",
        required=True,
        help="Path to the original project's run_results.json",
    )
    parser.add_argument(
        "--trained-models-bundle",
        default="useful_files/tuva_analysis/for_materialization_tuning/trained_models/original_0415_0333_trained_models_bundle.joblib",
        help="Path to the trained models bundle (joblib) for predicting cardinality of new models",
    )
    parser.add_argument(
        "--baseline-cardinality",
        default="useful_files/tuva_analysis/0826_baseline_cardinality.json",
        help="Path to baseline cardinality JSON for sources/seeds",
    )
    parser.add_argument(
        "--output-features",
        default=None,
        help="Override output feature file path",
    )
    parser.add_argument(
        "--output-graph",
        default=None,
        help="Override output graph file path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_filename = args.base_filename

    dbt_result_folder = f"useful_files/tuva_dbt_run_official_history/{base_filename}"
    MANIFEST_PATH = f"{dbt_result_folder}/manifest.json"

    output_feature_file = (
        args.output_features
        or f"useful_files/tuva_analysis/for_materialization_tuning/extracted_features_for_training/{base_filename}_extracted_features_for_training.json"
    )
    output_graph_file = (
        args.output_graph
        or f"useful_files/tuva_analysis/for_materialization_tuning/graph_files/{base_filename}_graph.json"
    )

    # ============= load original run_results.json for table cardinalities ===============
    with open(args.original_run_results) as f:
        original_run_results = json.load(f)

    original_table_stats = {}
    for result in original_run_results.get("results", []):
        unique_id = result.get("unique_id")
        adapter_response = result.get("adapter_response", {})
        if unique_id:
            if unique_id == "model.elementary.metadata":
                continue
            if adapter_response.get("code") == "CREATE TABLE":
                original_table_stats[unique_id] = {
                    "rows_affected": adapter_response.get("rows_affected", 0),
                    "slot_ms": adapter_response.get("slot_ms", 0),
                    "bytes_processed": adapter_response.get("bytes_processed", 0),
                }

    # ============= load trained models bundle for cardinality prediction =============

    with open(args.trained_models_bundle, "rb") as f:
        bundle = joblib.load(f)
    card_dv = bundle["cardinality"]["huber_all"]["dv"]
    card_pipe = bundle["cardinality"]["huber_all"]["pipe"]

    # ============= load baseline cardinality for sources/seeds ===============
    with open(args.baseline_cardinality) as f:
        baseline_cardinality_dict = json.load(f)

    # ============= load refactored manifest.json ===============
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # ============= build DAG and extract features from refactored manifest ===============
    G = nx.DiGraph()
    materialization = {}
    time_metrics = {}
    cardinality_metrics = {}

    nodes = manifest["nodes"]
    for uid, node in nodes.items():
        mat = node.get("config", {}).get("materialized", "").lower()
        if mat in ["test", "incremental", "seed"]:
            continue
        if uid.startswith("operation."):
            continue
        if "model.elementary." in uid or uid == "model.the_tuva_project.data_quality__testing_summary":
            continue

        raw_code = node.get("raw_code", "")
        for dep in node.get("depends_on", {}).get("nodes", []):
            dep_name = dep.split(".")[-1]
            w = count_refs(raw_code, dep_name)
            G.add_edge(dep, uid, weight=(w if w > 0 else 1))

        materialization[uid] = mat
        compiled_sql = node.get("compiled_code", "")

        time_metrics[uid] = analyze_sql_features_for_slot_time(compiled_sql) if compiled_sql else {}
        cardinality_metrics[uid] = analyze_sql_features_for_cardinality(compiled_sql) if compiled_sql else {}

    # ============= build table_stats from original, filtered to refactored manifest ===============
    table_stats_dict = {}
    for uid, stats in original_table_stats.items():
        if uid in materialization:
            table_stats_dict[uid] = stats

    # merge with baseline cardinality (sources and seeds — unchanged by refactoring)
    all_table_stat_dict = {**table_stats_dict, **baseline_cardinality_dict}

    print(f"Refactored manifest models: {len(materialization)}")
    print(f"Original table stats matched: {len(table_stats_dict)}")
    new_models = [uid for uid in materialization if uid not in original_table_stats and uid not in baseline_cardinality_dict]
    print(f"New models (no original cardinality): {len(new_models)}")
    for uid in new_models:
        print(f"  - {uid}")

    # ============= topological expansion + cardinality prediction for new models ===============
    topo = list(nx.topological_sort(G))

    expanded_time = deepcopy(time_metrics)
    expanded_cardinality = deepcopy(cardinality_metrics)
    sum_parents_cardinality_dict = defaultdict(int)

    # Initialize with known cardinalities; new models will be predicted below
    current_predicted_cardinality = {
        k: v.get("rows_affected", 0) for k, v in all_table_stat_dict.items()
    }

    for u in topo:
        # For new TABLE models not in all_table_stat_dict, predict cardinality
        # so that children receive correct sum_parents_cardinality.
        # Views don't need cardinality prediction — they propagate parent cardinality.
        if u in materialization and materialization.get(u) == "table" and u not in all_table_stat_dict:
            card_feat = expanded_cardinality.get(u, {}).copy()
            card_feat["sum_parents_cardinality_log1p"] = np.log1p(
                sum_parents_cardinality_dict.get(u, 0)
            )
            pred_log = card_pipe.predict(card_dv.transform(card_feat))
            current_predicted_cardinality[u] = max(0, np.expm1(pred_log)[0])
            print(f"  Predicted cardinality for {u}: {current_predicted_cardinality[u]:.0f}")

        # Expand features through views
        if u in materialization and materialization.get(u) not in ["table"]:
            for v in G.successors(u):
                w = G[u][v].get("weight", 1)
                add_scaled_inplace(expanded_time[v], expanded_time[u], w)
                add_scaled_inplace(expanded_cardinality[v], expanded_cardinality[u], w)

        # Propagate cardinality to children using current_predicted_cardinality
        if materialization.get(u) in ["table"] or u not in materialization:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += (
                    current_predicted_cardinality.get(u, 0)
                    * G[u][v].get("weight", 1)
                )
        else:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += (
                    sum_parents_cardinality_dict.get(u, 0)
                    * G[u][v].get("weight", 1)
                )

    # Set sum_parents_cardinality_log1p for all models
    for k in expanded_cardinality:
        spc = np.log1p(sum_parents_cardinality_dict.get(k, 0))
        expanded_time[k]["sum_parents_cardinality_log1p"] = spc
        expanded_cardinality[k]["sum_parents_cardinality_log1p"] = spc

    # Save predicted cardinalities for new table models into both table_stats
    # and all_table_stats so downstream scripts can reuse them directly
    for uid in new_models:
        if uid in current_predicted_cardinality and materialization.get(uid) == "table":
            entry = {
                "rows_affected": current_predicted_cardinality[uid],
                "predicted": True,
            }
            table_stats_dict[uid] = entry
            all_table_stat_dict[uid] = entry

    # ============= write outputs ===============
    os.makedirs(os.path.dirname(output_feature_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_graph_file), exist_ok=True)

    with open(output_feature_file, "w") as f:
        json.dump(
            {
                "expanded_time_features": expanded_time,
                "expanded_cardinality_features": expanded_cardinality,
                "table_stats": table_stats_dict,
                "all_table_stats": all_table_stat_dict,
                "materialization": materialization,
                "time_features_per_model": time_metrics,
                "cardinality_features_per_model": cardinality_metrics,
            },
            f,
            indent=2,
        )

    data = json_graph.node_link_data(G)
    with open(output_graph_file, "w") as f:
        json.dump(data, f)

    print(f"\nFeatures written to {output_feature_file}")
    print(f"Graph written to {output_graph_file}")


if __name__ == "__main__":
    main()
