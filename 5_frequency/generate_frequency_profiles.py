"""
Generate frequency profiles for cost-weighted materialization optimization.

Each profile assigns a cost multiplier (runs per month) to every node in the DAG
based on upstream data-change frequency and downstream use frequency.

Usage:
    python generate_frequency_profiles.py \
        --manifest-path useful_files/tuva_dbt_run_official_history/original_0211_0752/manifest.json \
        --num-scenarios 3 \
        --output-dir freq_profiles/
"""

import os
import json
import random
import argparse
from collections import defaultdict, deque

from utility.analyze import build_parent_child_maps, load_manifest


FREQ_OPTIONS = [1, 2, 3]  # 1: weekly, 2: daily, 3: hourly
FREQ_COST_MAP = {1: 1, 2: 7, 3: 168}  # cost multiplier (runs per month)

# Connectivity threshold: roots with >= this many descendants are "high-reach"
# and should not be assigned hourly (to avoid flooding the DAG via max-propagation).
ROOT_REACH_THRESHOLD = 50

# Predefined scenario settings with connectivity-aware root assignment.
# High-reach roots (large tables like medical_claim, terminology) get daily/weekly.
# Low-reach roots (specialty tables like ed_classification, ndc) can be hourly.
#                                    [weekly, daily, hourly]
SCENARIO_SETTINGS = [
    {  # Scenario 0: "Batch-heavy" — most high-reach roots daily, ~5% hourly
        "name": "batch_heavy",
        "high_root_ratios": [0.10, 0.85, 0.05],
        "low_root_ratios":  [0.00, 0.30, 0.70],
        "leaf_ratios":      [0.05, 0.15, 0.80],
    },
    {  # Scenario 1: "Moderate mismatch" — more weekly, ~5% hourly
        "name": "moderate_mismatch",
        "high_root_ratios": [0.25, 0.70, 0.05],
        "low_root_ratios":  [0.00, 0.15, 0.85],
        "leaf_ratios":      [0.05, 0.20, 0.75],
    },
    {  # Scenario 2: "Mixed cadence" — half weekly, ~5% hourly
        "name": "mixed_cadence",
        "high_root_ratios": [0.40, 0.55, 0.05],
        "low_root_ratios":  [0.00, 0.00, 1.00],
        "leaf_ratios":      [0.10, 0.25, 0.65],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate frequency profiles for cost-weighted optimization."
    )
    parser.add_argument(
        "--manifest-path", required=True,
        help="Path to dbt manifest.json",
    )
    parser.add_argument(
        "--num-scenarios", type=int, default=3,
        help="Number of frequency profiles to generate (default: 3)",
    )
    parser.add_argument(
        "--output-dir", default="freq_profiles",
        help="Output directory for profile JSONs (default: freq_profiles/)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (scenario i uses seed+i). Omit for non-deterministic.",
    )
    parser.add_argument(
        "--settings-json", default=None,
        help="Path to a JSON file with custom scenario settings "
             "(list of {name, root_ratios, leaf_ratios}). "
             "If omitted, uses built-in SCENARIO_SETTINGS.",
    )
    return parser.parse_args()


def assign_frequency_to_nodes(nodes, frequencies, ratios, rng):
    """Randomly assign frequencies to nodes according to the given ratios."""
    shuffled = nodes[:]
    rng.shuffle(shuffled)

    n = len(nodes)
    counts = [int(n * r) for r in ratios]
    remainder = n - sum(counts)
    # Add rounding remainder to the largest-ratio bucket (not always last,
    # which could accidentally assign hourly when hourly ratio is 0%).
    max_idx = max(range(len(ratios)), key=lambda i: ratios[i])
    counts[max_idx] += remainder

    result = {}
    start = 0
    for freq, cnt in zip(frequencies, counts):
        for node in shuffled[start:start + cnt]:
            result[node] = freq
        start += cnt
    return result


def count_descendants(root, child_map):
    """Count the number of descendants reachable from a root via BFS."""
    visited = set()
    q = deque([root])
    while q:
        n = q.popleft()
        for c in child_map.get(n, []):
            if c not in visited:
                visited.add(c)
                q.append(c)
    return len(visited)


def assign_frequency_connectivity_aware(root_nodes, child_map, threshold,
                                         high_ratios, low_ratios, rng,
                                         top_k_protected=1):
    """Assign root frequencies based on DAG connectivity.

    High-reach roots (>= threshold descendants) get high_ratios (typically low hourly %).
    Low-reach roots (< threshold descendants) get low_ratios (can be hourly).

    The top_k_protected highest-connectivity roots are explicitly excluded from
    hourly assignment (forced to daily) to prevent them from flooding the DAG
    via max-propagation.
    """
    root_desc = [(r, count_descendants(r, child_map)) for r in root_nodes]
    root_desc.sort(key=lambda x: -x[1])

    protected = {r for r, _ in root_desc[:top_k_protected]}
    high_reach = [r for r in root_nodes if count_descendants(r, child_map) >= threshold]
    low_reach = [r for r in root_nodes if count_descendants(r, child_map) < threshold]

    result = {}
    result.update(assign_frequency_to_nodes(high_reach, FREQ_OPTIONS, high_ratios, rng))
    result.update(assign_frequency_to_nodes(low_reach, FREQ_OPTIONS, low_ratios, rng))

    # Force protected roots to daily (override any hourly assignment)
    for r in protected:
        if result.get(r) == 3:
            result[r] = 2

    return result


def propagate_downstream(root_freq, child_map):
    """Propagate data-changing frequency from roots downstream (max)."""
    freq = defaultdict(int)
    q = deque()
    for node, f in root_freq.items():
        freq[node] = max(freq[node], f)
        q.append((node, f))
    while q:
        node, f = q.popleft()
        for child in child_map.get(node, []):
            if f > freq[child]:
                freq[child] = f
                q.append((child, f))
    return dict(freq)


def propagate_upstream(leaf_freq, parent_map):
    """Propagate use frequency from leaves upstream (max)."""
    freq = defaultdict(int)
    q = deque()
    for node, f in leaf_freq.items():
        freq[node] = max(freq[node], f)
        q.append((node, f))
    while q:
        node, f = q.popleft()
        for parent in parent_map.get(node, []):
            if f > freq[parent]:
                freq[parent] = f
                q.append((parent, f))
    return dict(freq)


def propagate_to_cost_multipliers(root_freq, leaf_freq, parent_map, child_map):
    """Propagate root/leaf frequencies through a DAG and convert to cost multipliers.

    Args:
        root_freq: {node_uid: freq_level} for root nodes (1=weekly, 2=daily, 3=hourly)
        leaf_freq: {node_uid: freq_level} for leaf nodes
        parent_map: {node: set_of_parents} for the DAG
        child_map: {node: set_of_children} for the DAG

    Returns:
        {node_uid: cost_multiplier} for all nodes with non-zero real update frequency
    """
    data_changing = propagate_downstream(root_freq, child_map)
    use_freq = propagate_upstream(leaf_freq, parent_map)

    all_nodes = set(data_changing.keys()) | set(use_freq.keys())

    node_frequencies = {}
    for node in all_nodes:
        dc = data_changing.get(node, 0)
        uf = use_freq.get(node, 0)
        real_update = min(dc, uf) if dc > 0 and uf > 0 else 0
        if real_update > 0:
            node_frequencies[node] = FREQ_COST_MAP[real_update]

    return node_frequencies


def generate_one_profile(parent_map, child_map, root_nodes, leaf_nodes,
                         high_root_ratios, low_root_ratios, leaf_ratios, rng,
                         threshold=ROOT_REACH_THRESHOLD):
    """Generate a single frequency profile with connectivity-aware root assignment.

    Returns:
        (node_frequencies, use_frequencies, root_freq, leaf_freq)
        - node_frequencies: real_update = min(data_changing, use) cost multipliers
        - use_frequencies: over-scheduled = use_frequency cost multipliers
        - root_freq, leaf_freq: raw assignments before propagation
    """
    root_freq = assign_frequency_connectivity_aware(
        root_nodes, child_map, threshold,
        high_root_ratios, low_root_ratios, rng,
    )
    leaf_freq = assign_frequency_to_nodes(
        leaf_nodes, FREQ_OPTIONS, leaf_ratios, rng
    )

    node_frequencies = propagate_to_cost_multipliers(
        root_freq, leaf_freq, parent_map, child_map
    )

    # use_frequencies: the over-scheduled rate (how often consumers need data,
    # ignoring whether upstream data actually changed)
    use_freq_raw = propagate_upstream(leaf_freq, parent_map)
    use_frequencies = {
        node: FREQ_COST_MAP[freq]
        for node, freq in use_freq_raw.items()
        if freq > 0
    }

    return node_frequencies, use_frequencies, root_freq, leaf_freq


def main():
    args = parse_args()

    manifest = load_manifest(args.manifest_path)
    parent_map, child_map = build_parent_child_maps(manifest)

    # Identify root and leaf nodes (same logic as frequency.py)
    root_nodes = []
    leaf_nodes = []
    for k, v in parent_map.items():
        if len(v) == 0 and len(child_map.get(k, [])) > 0:
            if k.startswith("model.elementary."):
                continue
            root_nodes.append(k)

    for k, v in child_map.items():
        if len(v) == 0 and len(parent_map.get(k, [])) > 0:
            if k.startswith("test.") or k.startswith("model.elementary."):
                continue
            leaf_nodes.append(k)

    print(f"Root nodes: {len(root_nodes)}")
    print(f"Leaf nodes: {len(leaf_nodes)}")

    # Load or select scenario settings
    if args.settings_json:
        with open(args.settings_json) as f:
            settings = json.load(f)
    else:
        settings = SCENARIO_SETTINGS

    num_scenarios = min(args.num_scenarios, len(settings))
    if args.num_scenarios > len(settings):
        print(f"WARNING: requested {args.num_scenarios} scenarios but only {len(settings)} settings available. Using {num_scenarios}.")

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(num_scenarios):
        s = settings[i]
        high_root_ratios = s["high_root_ratios"]
        low_root_ratios = s["low_root_ratios"]
        leaf_ratios = s["leaf_ratios"]
        scenario_name = s.get("name", f"scenario_{i}")

        seed_i = (args.seed + i) if args.seed is not None else None
        rng = random.Random(seed_i)

        node_frequencies, use_frequencies, root_freq, leaf_freq = generate_one_profile(
            parent_map, child_map, root_nodes, leaf_nodes,
            high_root_ratios, low_root_ratios, leaf_ratios, rng,
        )

        # Compute overall root frequency distribution
        root_freq_dist = defaultdict(int)
        for v in root_freq.values():
            root_freq_dist[v] += 1
        overall_root_ratios = [
            root_freq_dist.get(f, 0) / len(root_nodes) for f in FREQ_OPTIONS
        ]

        profile = {
            "metadata": {
                "scenario_name": f"scenario_{i}",
                "scenario_index": i,
                "setting_name": scenario_name,
                "high_root_ratios": high_root_ratios,
                "low_root_ratios": low_root_ratios,
                "overall_root_ratios": [round(r, 3) for r in overall_root_ratios],
                "leaf_ratios": leaf_ratios,
                "seed": seed_i,
                "num_root_nodes": len(root_nodes),
                "num_leaf_nodes": len(leaf_nodes),
                "num_nodes_with_frequency": len(node_frequencies),
            },
            "freq_cost_map": {str(k): v for k, v in FREQ_COST_MAP.items()},
            "root_frequencies": root_freq,
            "leaf_frequencies": leaf_freq,
            "node_frequencies": node_frequencies,
            "use_frequencies": use_frequencies,
        }

        output_path = os.path.join(args.output_dir, f"scenario_{i}.json")
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)

        # Summary stats
        freq_counts = defaultdict(int)
        for v in node_frequencies.values():
            freq_counts[v] += 1
        print(
            f"Scenario {i} ({scenario_name}): {len(node_frequencies)} nodes — "
            f"overall_root=[{overall_root_ratios[0]:.2f},{overall_root_ratios[1]:.2f},{overall_root_ratios[2]:.2f}] "
            f"leaf={leaf_ratios} — "
            f"weekly({freq_counts.get(1, 0)}) "
            f"daily({freq_counts.get(7, 0)}) "
            f"hourly({freq_counts.get(168, 0)})"
        )

    print(f"\n{num_scenarios} profiles written to {args.output_dir}/")


if __name__ == "__main__":
    main()
