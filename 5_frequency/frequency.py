import json
import random
import networkx as nx
from collections import defaultdict, deque
from utility.analyze import build_parent_child_maps

base_filename = "original_0211_0752"
dbt_result_folder = f"useful_files/tuva_dbt_run_official_history/{base_filename}"
MANIFEST_PATH = f"{dbt_result_folder}/manifest.json"
RUN_RESULTS_PATH = f"{dbt_result_folder}/run_results.json"

# ============= handle run_results.json ===============
with open(RUN_RESULTS_PATH) as f:
    run_results = json.load(f)

table_stats_dict = {}
for result in run_results.get("results", []):
    unique_id = result.get("unique_id")
    adapter_response = result.get("adapter_response", {})
    if unique_id:
        # if unique_id == 'model.elementary.metadata':
        #     continue
        
        # slot_ms, rows_affected, bytes_processed, etc.
        if adapter_response.get("code") == "CREATE TABLE":
            table_stats_dict[unique_id] = {
                "rows_affected": adapter_response.get("rows_affected", 0),
                "slot_ms": adapter_response.get("slot_ms", 0),
                "bytes_processed": adapter_response.get("bytes_processed", 0),
            }

# ============= handle manifest.json ===============
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
nodes = manifest["nodes"]

root_nodes = []
leaf_nodes = []
parent_map, child_map = build_parent_child_maps(manifest)
for k, v in parent_map.items():
    if (len(v) == 0) and (len(child_map.get(k, [])) > 0):
        if k.startswith("model.elementary."):
            continue
        root_nodes.append(k)
        # print(f"Root node: {k}")
        
for k, v in child_map.items():
    if (len(v) == 0) and (len(parent_map.get(k, [])) > 0):
        if k.startswith("test.") or k.startswith("model.elementary."):
            continue
        leaf_nodes.append(k)
        # print(f"Leaf node: {k}")

print(f"Total root nodes: {len(root_nodes)}")
print(f"Total leaf nodes: {len(leaf_nodes)}")


use_frequency_options = [1, 2, 3] # 1: weekly, 2: daily, 3: hourly
freq_cost_dict = {1: 1, 2: 7, 3: 168} # cost per month for each frequency


# write a function to assign the frequency to a list of nodes
def assign_frequency_to_nodes(nodes, frequency, ratio):
    """
    Randomly assign frequencies to all nodes according to the given ratios.
    :param nodes: List of nodes
    :param frequency: List of frequency values, e.g. [1, 2, 3]
    :param ratio: List of ratios, e.g. [0.1, 0.7, 0.2]
    :param seed: Optional random seed
    :return: Dictionary of node -> frequency
    """
    # use current time as seed
    # random.seed(42)
    random.seed()
    shuffled_nodes = nodes[:]
    random.shuffle(shuffled_nodes)

    n = len(nodes)
    counts = [int(n * r) for r in ratio]

    # fix rounding issue so total count matches n
    counts[-1] += n - sum(counts)
    node_to_frequency = {}
    start = 0
    for freq, cnt in zip(frequency, counts):
        for node in shuffled_nodes[start:start + cnt]:
            node_to_frequency[node] = freq
        start += cnt
    return node_to_frequency



root_frequency_dict = assign_frequency_to_nodes(root_nodes, use_frequency_options, [0.05, 0.85, 0.1])
leaf_frequency_dict = assign_frequency_to_nodes(leaf_nodes, use_frequency_options, [0.05, 0.3, 0.65])

def propagate_downstream(root_frequency_dict, child_map):
    """
    Propagate root frequencies from upstream to downstream.
    Each node keeps the highest propagated frequency reaching it.
    """
    data_changing_frequency = defaultdict(int)
    q = deque()

    for node, freq in root_frequency_dict.items():
        data_changing_frequency[node] = max(data_changing_frequency[node], freq)
        q.append((node, freq))

    while q:
        node, freq = q.popleft()
        for child in child_map.get(node, []):
            if freq > data_changing_frequency[child]:
                data_changing_frequency[child] = freq
                q.append((child, freq))

    return dict(data_changing_frequency)


def propagate_upstream(leaf_frequency_dict, parent_map):
    """
    Propagate leaf frequencies from downstream to upstream.
    Each node keeps the highest propagated frequency reaching it.
    """
    use_frequency = defaultdict(int)
    q = deque()

    for node, freq in leaf_frequency_dict.items():
        use_frequency[node] = max(use_frequency[node], freq)
        q.append((node, freq))

    while q:
        node, freq = q.popleft()
        for parent in parent_map.get(node, []):
            if freq > use_frequency[parent]:
                use_frequency[parent] = freq
                q.append((parent, freq))

    return dict(use_frequency)


# 1. propagate data-changing frequency: upstream -> downstream
data_changing_frequency_dict = propagate_downstream(root_frequency_dict, child_map)

# 2. propagate use frequency: downstream -> upstream
use_frequency_dict = propagate_upstream(leaf_frequency_dict, parent_map)

# 3. collect all nodes in the graph
all_nodes = set(data_changing_frequency_dict.keys()).union(set(use_frequency_dict.keys()))

# 4. for each node, compute real update frequency = min(data changing, use)
final_frequency_info = {}
for node in all_nodes:
    data_freq = data_changing_frequency_dict.get(node, 0)
    use_freq = use_frequency_dict.get(node, 0)

    # lower of the pair
    real_update_freq = min(data_freq, use_freq) if data_freq > 0 and use_freq > 0 else 0

    final_frequency_info[node] = {
        "data_changing_frequency": data_freq,
        "use_frequency": use_freq,
        "real_update_frequency": real_update_freq,
    }

# remove nodes with zero real update frequency
final_frequency_info = {node: info for node, info in final_frequency_info.items() if info["real_update_frequency"] > 0}
print(f"Total nodes with non-zero real update frequency: {len(final_frequency_info)}")

# for node, info in final_frequency_info.items():
#     print(
#         f"{node}: "
#         f"data_changing_frequency={info['data_changing_frequency']}, "
#         f"use_frequency={info['use_frequency']}, "
#         f"real_update_frequency={info['real_update_frequency']}"
#     )

expected_cost = 0
actual_cost = 0
cost_saved = 0

for k, v in final_frequency_info.items():
    if k not in table_stats_dict:
        continue
    use_freq = v["use_frequency"]
    update_freq = v["real_update_frequency"]
    expected_cost += table_stats_dict[k]["slot_ms"] * freq_cost_dict[use_freq]
    actual_cost += table_stats_dict[k]["slot_ms"] * freq_cost_dict[update_freq]
    if use_freq != update_freq:
        print(f"Node {k}: use_freq={use_freq}, update_freq={update_freq}, slot_ms={table_stats_dict[k]['slot_ms']}")
        
cost_saved = expected_cost - actual_cost
print(f"Expected cost: {expected_cost}, Actual cost: {actual_cost}, Cost saved: {cost_saved}")
# print ratio
ratio_saved = cost_saved / expected_cost if expected_cost > 0 else 0
print(f"Cost saved ratio: {ratio_saved:.2%}")




def find_split_candidates(parent_map, final_frequency_info):
    """
    Find nodes where some parents change slower than the node's real update frequency.
    These are candidates for splitting: extract the slow-changing parent dependencies
    into a separate precomputed model so they don't get recomputed at the higher frequency.
    """
    candidates = []
    for node, info in final_frequency_info.items():
        real_update_freq = info["real_update_frequency"]
        if real_update_freq == 0:
            continue

        parents = parent_map.get(node, [])
        if len(parents) < 2:
            continue

        fast_parents = []
        slow_parents = []
        for p in parents:
            p_info = final_frequency_info.get(p)
            if p_info is None:
                continue
            p_data_freq = p_info["data_changing_frequency"]
            if p_data_freq < real_update_freq:
                slow_parents.append((p, p_data_freq))
            else:
                fast_parents.append((p, p_data_freq))

        # only a candidate if there's at least one slow and one fast parent
        if slow_parents and fast_parents:
            candidates.append({
                "node": node,
                "real_update_frequency": real_update_freq,
                "fast_parents": fast_parents,
                "slow_parents": slow_parents,
            })

    return candidates

candidates = find_split_candidates(parent_map, final_frequency_info)
print(f"Total split candidates found: {len(candidates)}")
# for candidate in candidates:
#     print(f"Node {candidate['node']} (real update freq={candidate['real_update_frequency']}):")
#     print(f"  Fast parents: {[f'{p[0]}(data_freq={p[1]})' for p in candidate['fast_parents']]}")
#     print(f"  Slow parents: {[f'{p[0]}(data_freq={p[1]})' for p in candidate['slow_parents']]}")