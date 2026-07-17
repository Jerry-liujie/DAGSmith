import os
import argparse
import json
import numpy as np
import joblib
import networkx as nx
from networkx.readwrite import json_graph
from collections import defaultdict
from copy import deepcopy
from utility.analyze import add_scaled_inplace, solve_slot_time_ilp, summarize_iteration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative local linearization for materialization optimization."
    )
    parser.add_argument(
        "base_filename",
        help="Base filename for the variant "
             "(resolves to useful_files/tuva_dbt_run_official_history/{base_filename}/)",
    )
    parser.add_argument(
        "--trained-models-bundle",
        default="useful_files/tuva_analysis/for_materialization_tuning/trained_models/original_0415_0333_trained_models_bundle.joblib",
        help="Path to the trained models bundle (joblib)",
    )
    return parser.parse_args()


args = parse_args()
base_filename = args.base_filename

dbt_result_folder = f"useful_files/tuva_dbt_run_official_history/{base_filename}"
MANIFEST_PATH = f"{dbt_result_folder}/manifest.json"

FEATURES_JSON_FILE_PATH = f"useful_files/tuva_analysis/for_materialization_tuning/extracted_features_for_training/{base_filename}_extracted_features_for_training.json"
DEPENDENCY_GRAPH_PATH = f"useful_files/tuva_analysis/for_materialization_tuning/graph_files/{base_filename}_graph.json"
NEW_MATERIALIZATION_OUTPUT_PATH = f"useful_files/tuva_analysis/for_materialization_tuning/new_materializations/{base_filename}_new_materialization.json"

TRAINED_MODELS_BUNDLE_PATH = args.trained_models_bundle

# ============= handle manifest.json ===============
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
nodes = manifest.get("nodes", {})

# ============= load features ===============
with open(FEATURES_JSON_FILE_PATH) as f:
    features = json.load(f)

    time_features = features.get("time_features_per_model", {})
    cardinality_features = features.get("cardinality_features_per_model", {})
    table_stats = features.get("table_stats", {})
    all_table_stats = features.get("all_table_stats", {})
    materialization = features.get("materialization", {})

# ============= load Huber regression models ===============
with open(TRAINED_MODELS_BUNDLE_PATH, "rb") as f:
    bundle = joblib.load(f)
    card_dv = bundle["cardinality"]["huber_all"]["dv"]
    card_pipe = bundle["cardinality"]["huber_all"]["pipe"]
    time_dv = bundle["slot_time"]["huber"]["dv"]
    time_pipe = bundle["slot_time"]["huber"]["pipe"]

# ============== load dependency graph ===============
with open(DEPENDENCY_GRAPH_PATH) as f:
    data = json.load(f)
G = json_graph.node_link_graph(data)
topo = list(nx.topological_sort(G))

targets = [n for n in G if G.out_degree(n) == 0 and G.in_degree(n) > 0]
# Seeds/sources (not in materialization dict) must always be tables in the ILP
fixed_as_table = [n for n in G if n not in materialization]
all_targets = list(set(targets + fixed_as_table))
edges = list(G.edges())


MAX_ITERATIONS = 30
FANOUT_THRESHOLD = 20  # nodes with out_degree > this cannot be flipped to view
current_iteration = 0

current_table_cardinality = {k: v.get("rows_affected", 0) for k, v in all_table_stats.items()}

original_materialization = materialization.copy()
prev_materialization = materialization.copy()
history = {frozenset(materialization.items())}

# Track per-node flip count for diagnostics
node_flip_count = defaultdict(int)

# Fix 3: Identify high-fanout nodes that should not be flipped to view
high_fanout_nodes = {n for n in G if G.out_degree(n) > FANOUT_THRESHOLD and n in materialization}
if high_fanout_nodes:
    print(f"High-fanout nodes (out_degree > {FANOUT_THRESHOLD}, protected from view flip):")
    for n in sorted(high_fanout_nodes):
        print(f"  {n.split('.')[-1]}: out_degree={G.out_degree(n)}")

# =============== core iterative local linearization =================
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Models in materialization dict: {len(materialization)}")
print(f"Seeds/sources (fixed as table): {len(fixed_as_table)}")
print(f"Leaf targets: {len(targets)}")

while True:
    current_iteration += 1
    print(f"Iteration {current_iteration}")

    sum_parents_cardinality_dict = defaultdict(int)

    expanded_time = deepcopy(time_features)
    expanded_cardinality = deepcopy(cardinality_features)
    for u in topo:
        if u in materialization and materialization.get(u) not in ["table"]:
            for v in G.successors(u):
                w = G[u][v].get("weight", 1)
                add_scaled_inplace(expanded_time[v], expanded_time[u], w)
                add_scaled_inplace(expanded_cardinality[v], expanded_cardinality[u], w)

        # table model or seed model (not in materialization dict)
        if materialization.get(u) in ["table"] or u not in materialization:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += current_table_cardinality.get(u, 0) * G[u][v].get("weight", 1)
        else:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += sum_parents_cardinality_dict.get(u, 0) * G[u][v].get("weight", 1)

    for k, v in expanded_cardinality.items():
        sum_parents_cardinality_log1p = np.log1p(sum_parents_cardinality_dict.get(k, 0))
        expanded_cardinality[k]["sum_parents_cardinality_log1p"] = sum_parents_cardinality_log1p


    current_predicted_cardinality = {}
    for k, v in all_table_stats.items():
        current_predicted_cardinality[k] = v.get("rows_affected", 0)

    for k, v in materialization.items():
        if k not in all_table_stats:
            pred_log = card_pipe.predict(card_dv.transform(expanded_cardinality[k]))
            current_predicted_cardinality[k] = max(0.0, np.expm1(pred_log)[0])


    # calculate B for every node
    # Clamp to prevent numerical overflow from out-of-distribution feature propagation.
    # expm1(20) ≈ 485M ms — well above any real BigQuery query.
    B_MAX = np.expm1(20.0)

    B = {}
    for k, v in materialization.items():
        if k not in G:
            continue
        this_time_feature = time_features[k].copy()
        sum_parents_cardinality = 0
        for p in G.predecessors(k):
            sum_parents_cardinality += current_predicted_cardinality.get(p, 0)
        this_time_feature["sum_parents_cardinality_log1p"] = np.log1p(sum_parents_cardinality)
        this_time = np.expm1(time_pipe.predict(time_dv.transform(this_time_feature)))[0]
        B[k] = min(max(0.0, this_time), B_MAX)


    # calculate U for every edge
    U = {}
    for k, v in materialization.items():
        if k not in G:
            continue

        for p in G.predecessors(k):
            # suppose that this p is a view
            this_time_feature = time_features[k].copy()
            if p in expanded_time:
                w = G[p][k].get("weight", 1)
                add_scaled_inplace(this_time_feature, expanded_time[p], w)

            this_sum_parents_cardinality = 0
            for pp in G.predecessors(k):
                if pp == p:
                    this_sum_parents_cardinality += sum_parents_cardinality_dict.get(pp, 0)
                else:
                    this_sum_parents_cardinality += current_predicted_cardinality.get(pp, 0)

            this_time_feature["sum_parents_cardinality_log1p"] = np.log1p(this_sum_parents_cardinality)
            this_time = np.expm1(time_pipe.predict(time_dv.transform(this_time_feature)))[0]
            this_time = min(max(0.0, this_time), B_MAX)
            U[(p, k)] = max(min(this_time - B[k], B_MAX), -B_MAX)

    t_ref = {}
    for k, v in B.items():
        if materialization.get(k) in ["table"]:
            t_ref[k] = 1
        else:
            t_ref[k] = 0
    # Seeds/sources are always tables — include in t_ref so they don't count as flips
    for n in fixed_as_table:
        t_ref[n] = 1

    # Fix 3: Force high-fanout nodes to remain tables in the ILP.
    protected_targets = list(all_targets)
    for n in high_fanout_nodes:
        if n not in protected_targets:
            protected_targets.append(n)

    try:
        t_sol, y_sol, obj, flips, status = solve_slot_time_ilp(
            B=B, U=U, edges=edges, targets=protected_targets,
            t_ref=t_ref, trust_region_L=3, flip_penalty=1.0,
            solver_name="highs"  # or "cbc"/"gurobi"
        )
    except Exception as e:
        print(f"  ILP solver failed: {e}")
        print(f"  Stopping at iteration {current_iteration}, using current materialization.")
        break

    num_flips = sum(flips.values()) if flips else 0
    num_tables = sum(1 for v in t_sol.values() if v == 1)
    num_views = sum(1 for v in t_sol.values() if v == 0)
    print(f"  ILP objective: {obj:,.2f}  status: {status}  flips: {num_flips}  tables: {num_tables}  views: {num_views}")

    # update materialization (skip models not in graph to preserve their original config)
    for k, v in materialization.items():
        if k not in G:
            continue
        if k in t_sol and t_sol[k] == 1:
            materialization[k] = "table"
        else:
            materialization[k] = "view"

    # Track per-node flips for diagnostics
    for k in materialization:
        if k in prev_materialization and materialization[k] != prev_materialization[k]:
            node_flip_count[k] += 1

    # update current table cardinality (preserve seed/source entries from all_table_stats)
    current_table_cardinality = {
        k: v.get("rows_affected", 0)
        for k, v in all_table_stats.items()
    }
    current_table_cardinality.update({
        k: current_predicted_cardinality[k]
        for k, v in materialization.items()
        if v == "table"
    })

    # check for oscillation (repeated materialization state)
    state = frozenset(materialization.items())
    if state in history:
        print(f"Oscillation detected at iteration {current_iteration} — stopping.")
        break
    history.add(state)

    if current_iteration >= MAX_ITERATIONS or sum(flips.values()) == 0:
        print(f"number of iterations: {current_iteration}")
        break

    # check for changes in materialization vs previous iteration
    if prev_materialization != materialization:
        print("Materialization changes detected:")
        cnt = 0
        for k, v in materialization.items():
            if prev_materialization.get(k) != v:
                cnt += 1
                print(f"  - {k}: {prev_materialization.get(k)} -> {v}")
        print(f"Total changes: {cnt}")

    prev_materialization = materialization.copy()

# Report oscillating nodes
oscillating = {k: c for k, c in node_flip_count.items() if c >= 3}
if oscillating:
    print(f"\nNote: {len(oscillating)} nodes oscillated 3+ times:")
    for k, c in sorted(oscillating.items(), key=lambda x: -x[1]):
        print(f"  {k.split('.')[-1]}: flipped {c} times")

# write new materialization to json file
os.makedirs(os.path.dirname(NEW_MATERIALIZATION_OUTPUT_PATH), exist_ok=True)
with open(NEW_MATERIALIZATION_OUTPUT_PATH, "w") as f:
    json.dump(materialization, f, indent=4)

print(f"\nNew materialization written to {NEW_MATERIALIZATION_OUTPUT_PATH}")
