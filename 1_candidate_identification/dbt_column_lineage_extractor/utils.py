from __future__ import annotations

import json
import os
import logging
import sys

import math
from collections import deque
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Set, Tuple, Any, Iterable
import sqlglot
import sqlglot.expressions as exp


def setup_logging():
    """
    Set up logging configuration.
    Returns:
        Logger object
    """
    logger = logging.getLogger("dbt_column_lineage")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def pretty_print_dict(dict_to_print):
    """
    Pretty print a dictionary as JSON.
    Logs the formatted JSON and also prints it directly for test compatibility.

    Args:
        dict_to_print: Dictionary to print

    Returns:
        Formatted JSON string
    """
    formatted_json = json.dumps(dict_to_print, indent=4)

    # Log using the logger
    logger = setup_logging()
    logger.info(formatted_json)

    # Also print directly for test compatibility
    print(formatted_json)

    return formatted_json


def write_dict_to_file(dict_to_write, file_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        json.dump(dict_to_write, file, indent=4)


def read_dict_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def find_potential_matches(lineage_data, model_name):
    """Find potential model matches based on partial name match."""
    model_name = model_name.lower()
    return [model for model in lineage_data.keys() if model_name in model.lower()]


def find_exact_name_matches(lineage_data, model_name):
    """Find models that match exactly by node name (after the resource type and package)."""
    model_name = model_name.lower()
    # Extract just the name part from full node paths like 'model.analytics.aicg__fact_user'
    return [model for model in lineage_data.keys() if model.lower().split(".")[-1] == model_name]

# ============================================================
# Helpers: identifiers, relation strings, fingerprints
# ============================================================

def id_for(parts: Iterable[Optional[str]]) -> str:
    return ".".join([p for p in parts if p])

def sanitize_relation(rel: str) -> str:
    if rel is None:
        return ""
    r = rel.replace("`", "").replace("[", "").replace("]", "").replace('"', "")
    # Collapse multiple separators and drop empties
    return ".".join([p for p in r.split(".") if p])

def ident_str(x):
    return x.name if isinstance(x, exp.Identifier) else (x or "")

def table_relation_str(t: exp.Table) -> str:
    # Build project.dataset.table from parts; strip alias/quotes
    cat = ident_str(t.args.get("catalog"))
    db  = ident_str(t.args.get("db"))
    th  = ident_str(t.args.get("this"))
    return sanitize_relation(id_for([cat, db, th]))

def col_fq(parts: List[exp.Identifier]) -> str:
    return id_for([p.name for p in parts])

def collect_columns(e: exp.Expression) -> Set[str]:
    cols = set()
    for c in e.find_all(exp.Column):
        cols.add(col_fq(c.parts))
    return cols
    
    
def set_edit_distance_substitution(a: set, b: set) -> int:
    """
    Edit distance between two same-size sets,
    where one substitution (remove + add) costs 1.
    """
    assert len(a) == len(b), "This definition assumes same-size sets."
    sym_diff_size = len(a.symmetric_difference(b))
    return sym_diff_size // 2


def anchor_dist(a: Set[str], b: Set[str]) -> Tuple[int, int]:
    common = len(a & b)
    dist = max(len(a), len(b)) - common
    return dist, common


def induced_edges(nodes: List[str], adj: Dict[str, Set[str]]) -> int:
        s = set(nodes)
        e = 0
        for u in nodes:
            for v in adj.get(u, []):
                if v in s and u < v:
                    e += 1
        return e

def k_core(nodes: List[str], k: int, adj: Dict[str, Set[str]]) -> List[str]:
    s = set(nodes)
    deg = {u: sum(1 for v in adj[u] if v in s) for u in s}
    dq = deque([u for u in s if deg[u] < k])
    while dq:
        u = dq.popleft()
        if u not in s:
            continue
        s.remove(u)
        for v in adj[u]:
            if v in s:
                deg[v] -= 1
                if deg[v] < k:
                    dq.append(v)
    return sorted(s)






def build_idf(sigs):
    df = defaultdict(int)
    N = len(sigs)
    for s in sigs:
        for tok in s.tokens.keys():
            df[tok] += 1
    idf = {}
    for tok, d in df.items():
        # smooth IDF
        idf[tok] = math.log((N + 1) / (d + 1)) + 1.0
    return idf


def weighted_jaccard(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        num = sum(min(a.get(k,0.0), b.get(k,0.0)) for k in keys)
        den = sum(max(a.get(k,0.0), b.get(k,0.0)) for k in keys)
        return (num/den) if den else 0.0


def weighted_jaccard_idf(a, b, idf):
    keys = set(a) | set(b)
    num = 0.0
    den = 0.0
    for k in keys:
        wa = a.get(k, 0.0) * idf.get(k, 1.0)
        wb = b.get(k, 0.0) * idf.get(k, 1.0)
        num += min(wa, wb)
        den += max(wa, wb)
    return (num / den) if den else 0.0


def is_expensive_anchor(tok: str) -> bool:
    return tok.startswith("join:") or tok.startswith("group_key:")


def shared_weight(a, b, idf):
    # sum of IDF weights over shared tokens
    return sum(idf.get(t, 1.0) for t in (set(a) & set(b)))


def connected_components(nodes: Set[str], adj: Dict[str, Set[str]]) -> List[List[str]]:
    visited = set()
    comps: List[List[str]] = []

    for n in nodes:
        if n in visited:
            continue
        q = deque([n])
        visited.add(n)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        comps.append(sorted(comp))

    comps.sort(key=len, reverse=True)
    return comps


def cc_sizes_from_topk_pairs(pairs, K):
    """
    pairs: list of (combined, sim, shared_w, sig_i, sig_j), sorted desc by combined
    K: number of top edges to include
    Returns: list of CC sizes (ints), considering only models that appear in edges.
    """
    adj = defaultdict(set)
    nodes = set()

    for _, _, _, a, b in pairs[:K]:
        u, v = a.uid, b.uid
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
        nodes.add(u)
        nodes.add(v)

    visited = set()
    sizes = []
    for n in nodes:
        if n in visited:
            continue
        q = deque([n])
        visited.add(n)
        cnt = 0
        while q:
            cur = q.popleft()
            cnt += 1
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        sizes.append(cnt)

    return sizes

def backbone_similarity(
    bb1,
    bb2,
    *,
    use_multiplicity: bool = True,
    eps: float = 1e-9
) -> float:
    """
    Jaccard similarity = |intersection| / |union|, common multiset, sum of common multiset counts

    If use_multiplicity=True: multiset Jaccard using min/max counts per token.
    Else: set Jaccard on token presence only.
    """
    
    def _bb_to_counter(bb) -> Counter:
        """Convert a frozen bb_key or Counter into a Counter(token -> multiplicity)."""
        if isinstance(bb, Counter):
            return bb
        # assume bb is bb_key: tuple of (token, count)
        return Counter(dict(bb))
    
    c1 = _bb_to_counter(bb1)
    c2 = _bb_to_counter(bb2)
    
    common = Counter()

    if not use_multiplicity:
        s1, s2 = set(c1.keys()), set(c2.keys())
        inter = len(s1 & s2)
        uni = len(s1 | s2)
        return inter / (uni + eps)

    # multiset (weighted) Jaccard
    keys = set(c1.keys()) | set(c2.keys())
    inter = 0
    uni = 0
    for k in keys:
        a = c1.get(k, 0)
        b = c2.get(k, 0)
        inter += min(a, b)
        uni += max(a, b)
        common[k] = min(a, b)

    return inter / (uni + eps), common, inter