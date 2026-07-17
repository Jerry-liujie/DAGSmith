import json
import argparse
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from google.cloud import bigquery
from utility.analyze import analyze_sql_features_for_slot_time,\
    analyze_sql_features_for_cardinality, count_refs, add_scaled_inplace
import numpy as np
from copy import deepcopy
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features for training from dbt manifest and run results."
    )
    parser.add_argument(
        "--base-filename",
        required=True,
        help="Base filename (resolves to useful_files/tuva_dbt_run_official_history/{base_filename}/)",
    )
    parser.add_argument(
        "--skeleton",
        required=True,
        help="Path to the skeleton JSON file",
    )
    return parser.parse_args()


args = parse_args()
base_filename = args.base_filename

dbt_result_folder = f"useful_files/tuva_dbt_run_official_history/{base_filename}"

output_feature_file = f"useful_files/tuva_analysis/for_materialization_tuning/extracted_features_for_training/{base_filename}_extracted_features_for_training.json"
output_graph_file = f"useful_files/tuva_analysis/for_materialization_tuning/graph_files/{base_filename}_graph.json"

MANIFEST_PATH = f"{dbt_result_folder}/manifest.json"
RUN_RESULTS_PATH = f"{dbt_result_folder}/run_results.json"

MODEL_SKELETON_PATH = args.skeleton

BASE_TABLE_CARDINALITY_STATS_PATH = "useful_files/tuva_analysis/0826_baseline_cardinality.json"


# ============= handle run_results.json ===============
with open(RUN_RESULTS_PATH) as f:
    run_results = json.load(f)

# only tables are training samples
table_stats_dict = {}
for result in run_results.get("results", []):
    unique_id = result.get("unique_id")
    adapter_response = result.get("adapter_response", {})
    if unique_id:
        if unique_id == 'model.elementary.metadata':
            continue
        
        # slot_ms, rows_affected, bytes_processed, etc.
        if adapter_response.get("code") == "CREATE TABLE":
            table_stats_dict[unique_id] = {
                "rows_affected": adapter_response.get("rows_affected", 0),
                "slot_ms": adapter_response.get("slot_ms", 0),
                "bytes_processed": adapter_response.get("bytes_processed", 0),
            }

# ============= handle model skeleton tokens stats =============
with open(MODEL_SKELETON_PATH, "r") as f:
    model_skeleton_dict = json.load(f)

# ============= handle baseline_cardinality.json ===============
with open(BASE_TABLE_CARDINALITY_STATS_PATH, "r") as f:
    baseline_cardinality_dict = json.load(f)

# merge two dicts
all_table_stat_dict = {**table_stats_dict, **baseline_cardinality_dict}

# ============= handle manifest.json ===============
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

G = nx.DiGraph()
# unique_id -> 'view' or 'table' (or others)
materialization = {}
time_metrics = {}
cardinality_metrics = {}


# PROJECT  = os.environ.get("DBT_BQ_PROJECT")
# LOCATION = "US"
# client = bigquery.Client(project=PROJECT, location=LOCATION)

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
        dep_name = dep.split('.')[-1]
        w = count_refs(raw_code, dep_name)
        G.add_edge(dep, uid, weight=(w if w > 0 else 1))
        
    materialization[uid] = mat
    compiled_sql = node.get("compiled_code", "")
    
    time_metrics[uid] = analyze_sql_features_for_slot_time(compiled_sql) if compiled_sql else {}
    cardinality_metrics[uid] = analyze_sql_features_for_cardinality(compiled_sql) if compiled_sql else {}

# Topologically sort and DP‐accumulate execution counts
topo = list(nx.topological_sort(G))

# deepcopy
expanded_time           = deepcopy(time_metrics)
expanded_cardinality    = deepcopy(cardinality_metrics)
expanded_skeleton       = deepcopy(model_skeleton_dict)    

sum_parents_cardinality_dict = defaultdict(int)

for u in topo:
    if u in materialization and materialization.get(u) not in ["table"]:
        for v in G.successors(u):
            w = G[u][v].get("weight", 1)
            # add *expanded* features of u, scaled by reference count, into v
            add_scaled_inplace(expanded_time[v], expanded_time[u], w)
            add_scaled_inplace(expanded_cardinality[v], expanded_cardinality[u], w)
            add_scaled_inplace(expanded_skeleton.get(v, {}), expanded_skeleton.get(u, {}), w)

    # table model or seed model (not in materialization dict)
    if materialization.get(u) in ["table"] or u not in materialization:
        for v in G.successors(u):
            sum_parents_cardinality_dict[v] += all_table_stat_dict.get(u, {}).get("rows_affected", 0) * G[u][v].get("weight", 1)
    else:
        for v in G.successors(u):
            sum_parents_cardinality_dict[v] += sum_parents_cardinality_dict.get(u, 0) * G[u][v].get("weight", 1)

for k, v in expanded_cardinality.items():
    sum_parents_cardinality_log1p = np.log1p(sum_parents_cardinality_dict.get(k, 0))
    expanded_time[k]["sum_parents_cardinality_log1p"] = sum_parents_cardinality_log1p
    expanded_cardinality[k]["sum_parents_cardinality_log1p"] = sum_parents_cardinality_log1p




# dump all info to a json file
with open(output_feature_file, "w") as f:
    json.dump({
        "expanded_time_features": expanded_time,
        "expanded_cardinality_features": expanded_cardinality,
        "expanded_skeleton_features": expanded_skeleton,
        "table_stats": table_stats_dict, # table models not including sources and seeds
        "all_table_stats": all_table_stat_dict, # including base tables (sources and seeds)
        "materialization": materialization,
        "time_features_per_model": time_metrics,
        "cardinality_features_per_model": cardinality_metrics,
        "skeleton_features_per_model": model_skeleton_dict,
    }, f, indent=2)

data = json_graph.node_link_data(G)
with open(output_graph_file, "w") as f:
    json.dump(data, f)

