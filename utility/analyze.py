from __future__ import annotations
import sys
import json
import re
import pandas as pd
from typing import Dict, Optional, Tuple, List, Iterable, Mapping, Set
from sqlglot import exp, parse_one
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, BadRequest
from io import BytesIO
from math import log1p
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, precision_recall_curve, f1_score
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.linear_model import HuberRegressor, ElasticNet, LogisticRegression
import scipy.sparse as sp



import pyomo.environ as pyo
from pathlib import Path
import tiktoken
import matplotlib.pyplot as plt


def analyze_sql_feature_basic(sql: str):
    """
    Parse the SQL and return metrics:
      - joins: number of JOIN clauses
      - aggs: number of group bys
      - windows: number of windows
    """
    try:
        tree = parse_one(sql, read="bigquery")
    except Exception as e:
        print(f"parse error: {e}")
        return None

    # join nodes
    num_joins = len(list(tree.find_all(exp.Join)))

    # group by nodes
    num_aggs = len(list(tree.find_all(exp.Group)))

    # window nodes
    num_windows = len(list(tree.find_all(exp.Window)))

    return {
        "joins": num_joins,
        "aggs": num_aggs,
        "windows": num_windows
    }

# ------------------------------------------------------------
# Utility: parse & extract common features used by both models
# ------------------------------------------------------------

def _extract_sql_features(sql: str):
    """
    Analyzes SQL and returns an execution-weighted feature set as a JSON string.
    Accounts for CTE reuse and distinguishes between join keys and predicates.
    """
    try:
        tree = parse_one(sql, read="bigquery")
    except Exception as e:
        return json.dumps({"error": f"Parse error: {str(e)}"})

    # Map CTE definitions for recursive lookup
    cte_defs = {cte.alias: cte.this for cte in tree.find_all(exp.CTE)}

    # Separate the main body from the WITH clause to avoid double-counting
    main_query = tree.copy()
    if main_query.find(exp.With):
        main_query.find(exp.With).pop()

    def get_base_metrics(node) -> Dict[str, int]:
        m = {
            "table_scans": 0,
            "joins_count": 0,
            "join_keys": 0,
            "non_equi_join_count": 0,
            "predicates_count": 0,
            "pred_eq": 0,
            "pred_range": 0,
            "pred_in": 0,
            "group_by_count": 0,
            "group_by_keys": 0,
            "window_funcs": 0,
            "aggs_total": 0,
            "unions": 0,
            "distinct_count": 0,
            "order_by_count": 0,
        }

        # ---------- helpers ----------

        def is_literal_like(e: exp.Expression) -> bool:
            """
            Treat constants and literal containers as literal-like.
            """
            return isinstance(e, (
                exp.Literal,
                exp.Boolean,
                exp.Null,
            ))

        def is_column_like(e: exp.Expression) -> bool:
            """
            Treat columns and simple deterministic expressions over columns as column-like.
            This makes things like DATE(col) = DATE(col2) still count as join-like.
            """
            if isinstance(e, exp.Column):
                return True

            # No columns anywhere -> not column-like
            has_col = any(True for _ in e.find_all(exp.Column))
            if not has_col:
                return False

            # If expression contains a subquery, treat it as not column-like
            if any(True for _ in e.find_all(exp.Subquery)):
                return False

            return True

        def has_literal_side(left: exp.Expression, right: exp.Expression) -> bool:
            return is_literal_like(left) or is_literal_like(right)

        def both_column_like(left: exp.Expression, right: exp.Expression) -> bool:
            return is_column_like(left) and is_column_like(right)

        def flatten_conjuncts(e: Optional[exp.Expression]) -> Iterable[exp.Expression]:
            """
            Break a AND b AND c into [a, b, c].
            """
            if e is None:
                return
            if isinstance(e, exp.And):
                yield from flatten_conjuncts(e.left)
                yield from flatten_conjuncts(e.right)
            else:
                yield e

        def classify_binary_condition(
            cond: exp.Expression,
            in_join_on: bool,
        ) -> None:
            """
            Update metrics for binary conditions such as =, !=, <, <=, >, >=.
            """
            if not isinstance(cond, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                return

            left = cond.left
            right = cond.right
            is_eq = isinstance(cond, exp.EQ)
            is_range = isinstance(cond, (exp.GT, exp.GTE, exp.LT, exp.LTE))
            is_non_eq_binary = isinstance(cond, (exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE))

            if in_join_on:
                # JOIN ... ON
                if both_column_like(left, right):
                    if is_eq:
                        m["join_keys"] += 1
                    else:
                        m["non_equi_join_count"] += 1
                else:
                    # Predicate disguised as join condition
                    m["predicates_count"] += 1
                    if is_eq:
                        m["pred_eq"] += 1
                    elif is_range or isinstance(cond, exp.NEQ):
                        m["pred_range"] += 1
            else:
                # WHERE
                if both_column_like(left, right):
                    # Join disguised as predicate
                    if is_eq:
                        m["join_keys"] += 1
                    else:
                        m["non_equi_join_count"] += 1
                else:
                    # Normal predicate
                    m["predicates_count"] += 1
                    if is_eq:
                        m["pred_eq"] += 1
                    elif is_range or isinstance(cond, exp.NEQ):
                        m["pred_range"] += 1

        def classify_in_condition(cond: exp.In, in_join_on: bool) -> None:
            """
            IN is usually a predicate, not a join key.
            """
            m["predicates_count"] += 1
            m["pred_in"] += 1

        def classify_between_condition(cond: exp.Between, in_join_on: bool) -> None:
            m["predicates_count"] += 1
            m["pred_range"] += 1

        def classify_like_condition(cond: exp.Expression, in_join_on: bool) -> None:
            m["predicates_count"] += 1
            m["pred_range"] += 1

        def classify_condition(cond: exp.Expression, in_join_on: bool) -> None:
            if isinstance(cond, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
                classify_binary_condition(cond, in_join_on)
            elif isinstance(cond, exp.In):
                classify_in_condition(cond, in_join_on)
            elif isinstance(cond, exp.Between):
                classify_between_condition(cond, in_join_on)
            elif isinstance(cond, (exp.Like, exp.ILike)):
                classify_like_condition(cond, in_join_on)
            # optionally support IS / IS NOT
            elif isinstance(cond, exp.Is):
                m["predicates_count"] += 1
                m["pred_eq"] += 1

        # ---------- 1. joins ----------
        joins = list(node.find_all(exp.Join))
        m["joins_count"] = len(joins)

        for j in joins:
            on = j.args.get("on")
            for cond in flatten_conjuncts(on):
                classify_condition(cond, in_join_on=True)

        # ---------- 2. where ----------
        for w in node.find_all(exp.Where):
            for cond in flatten_conjuncts(w.this):
                classify_condition(cond, in_join_on=False)

        # 3. Aggregations & Structs
        m["group_by_count"] = len(list(node.find_all(exp.Group)))
        m["group_by_keys"] = sum(len(g.expressions) for g in node.find_all(exp.Group))
        m["aggs_total"] = len(list(node.find_all(exp.AggFunc)))
        m["window_funcs"] = len(list(node.find_all(exp.Window)))
        m["unions"] = len(list(node.find_all(exp.Union)))
        m["distinct_count"] = len(list(node.find_all(exp.Distinct)))
        m["order_by_count"] = len(list(node.find_all(exp.Order)))

        # 4. Physical Scans (Excluding CTE aliases)
        m["table_scans"] = sum(1 for t in node.find_all(exp.Table) if t.name not in cte_defs)

        return m

    # Recursive logic with memoization to handle CTE re-use
    memo = {}

    def get_recursive_metrics(name, expression) -> Dict[str, int]:
        if name in memo: 
            return memo[name]
        
        current = get_base_metrics(expression)
        
        # Propagate costs from nested CTEs
        for table in expression.find_all(exp.Table):
            if table.name in cte_defs and table.name != name:
                nested_costs = get_recursive_metrics(table.name, cte_defs[table.name])
                for k, v in nested_costs.items():
                    current[k] += v
        
        memo[name] = current
        return current

    # Start tallying from the main query body
    final_feats = get_base_metrics(main_query)

    # Multiply CTE costs by reference count in the main body
    for table in main_query.find_all(exp.Table):
        if table.name in cte_defs:
            cte_costs = get_recursive_metrics(table.name, cte_defs[table.name])
            for k, v in cte_costs.items():
                final_feats[k] += v

    return tree, final_feats


bytes_scanned_features = {
    "table_scans",
    "pred_eq",
    "pred_range",
    "pred_in",
    "unions",
    "joins_count"
}

cardinality_features = {
    "joins_count",
    "join_keys",
    "non_equi_join_count",
    "pred_eq",
    "pred_range",
    "pred_in",
    "group_by_count",
    "group_by_keys",
    "distinct_count",
    "unions",
    "table_scans",
}

slot_time_features = {
    "table_scans",
    "joins_count",
    "join_keys",
    "non_equi_join_count",
    "pred_eq",
    "pred_range",
    "pred_in",
    "group_by_count",
    "group_by_keys",
    "window_funcs",
    "aggs_total",
    "unions",
    "distinct_count",
    "order_by_count",
}


# ------------------------------------------------------------
# Bytes_scanned features (rows_out)
# ------------------------------------------------------------
def analyze_sql_features_for_bytes_scanned(sql: str) -> Optional[Dict[str, int]]:
    tree, base = _extract_sql_features(sql)
    if tree is None:
        return None

    feats: Dict[str, int] = {}

    # Common subset relevant for cardinality
    for k in bytes_scanned_features:
        feats[k] = base.get(k, 0)
    return feats



# ------------------------------------------------------------
# Cardinality features (rows_out)
# ------------------------------------------------------------
def analyze_sql_features_for_cardinality(sql: str, dry_run_bytes: Optional[int] = None) -> Optional[Dict[str, int]]:
    tree, base = _extract_sql_features(sql)
    if tree is None:
        return None

    feats: Dict[str, int] = {}
    
    # Optional but very predictive
    if dry_run_bytes is not None:
        if dry_run_bytes < 0:
            feats["dry_run_bytes_log1p"] = dry_run_bytes
        else:
            feats["dry_run_bytes_log1p"] = int(log1p(int(dry_run_bytes)))

    # Common subset relevant for cardinality
    for k in cardinality_features:
        feats[k] = base.get(k, 0)
    return feats


# ------------------------------------------------------------
# Slot-time features (slot_ms)
# ------------------------------------------------------------
def analyze_sql_features_for_slot_time(sql: str) -> Optional[Dict[str, int]]:
    tree, base = _extract_sql_features(sql)
    if tree is None:
        return None

    feats: Dict[str, int] = {}

    # Common subset relevant for time
    for k in slot_time_features:
        feats[k] = base.get(k, 0)

    return feats




def get_row_count_for_table(client: bigquery.Client, project: str, dataset: str, table_id: str, table_type: str) -> int | None:
    """
    Return the row count for a BigQuery table or view.
    - For TABLE or MATERIALIZED_VIEW: read num_rows metadata.
    - For VIEW: run a COUNT(*) query.
    Returns None if the table/view is not found or if a query error occurs.
    """
    fq_table = f"{project}.{dataset}.{table_id}"
    try:
        tbl = client.get_table(fq_table)
    except NotFound:
        return None
    except Exception as e:
        print(f"[ERROR] Retrieving metadata for `{fq_table}`: {e}")
        return None

    ttype = table_type.upper()
    if ttype in ("TABLE", "MATERIALIZED_VIEW"):
        return tbl.num_rows
    elif ttype == "VIEW":
        sql = f"SELECT COUNT(*) AS cnt FROM `{fq_table}`"
        try:
            query_job = client.query(sql)
            result = query_job.result()
            for row in result:
                return int(row.cnt)
        except (BadRequest, Exception) as e:
            print(f"[ERROR] COUNT(*) failed for view `{fq_table}`: {e}")
            return None
    else:
        # Unexpected type; fall back to metadata if available
        return tbl.num_rows


def row_counts_for_datasets(project: str, datasets: List[str], location: str = "US") -> pd.DataFrame:
    """
    Returns a DataFrame listing each table or view in the specified BigQuery datasets
    along with its row count.

    :param project: GCP project ID
    :param datasets: List of dataset names within the project
    :param location: BigQuery location (default "US")
    """
    client = bigquery.Client(project=project, location=location)
    rows = []

    for dataset in datasets:
        dataset_ref = f"{project}.{dataset}"
        try:
            table_list = client.list_tables(dataset_ref, max_results=10000)
        except Exception as e:
            print(f"[ERROR] Unable to list tables for dataset `{dataset_ref}`: {e}")
            continue

        for table_item in table_list:
            table_type = table_item.table_type  # e.g., 'TABLE', 'VIEW', 'MATERIALIZED_VIEW', 'EXTERNAL'
            table_id = table_item.table_id

            # Fetch row count (tables and materialized views from metadata; views via COUNT(*))
            row_count = get_row_count_for_table(client, project, dataset, table_id, table_type)

            rows.append({
                "dataset": dataset,
                "table": table_id,
                "type": table_type,
                "row_count": row_count
            })

    df = pd.DataFrame(rows)
    return df


def extract_materializations(manifest_path):
    """
    Load manifest.json and return a list of (unique_id, materialized) for each model.
    """
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find manifest at {manifest_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    nodes = manifest.get('nodes', {})
    results = []
    for unique_id, node in nodes.items():
        # We only care about models
        if not unique_id.startswith('model.'):
            continue

        # The node['config']['materialized'] holds the materialization strategy
        materialized = node.get('config', {}).get('materialized', 'undefined')
        # Optionally, you can pretty-print the model's file path or alias instead of unique_id:
        #     model_name = node.get('path')  # e.g. "models/foo/bar.sql"
        #     model_name = node.get('alias') or node.get('name')
        results.append((unique_id, materialized))

    return results


def load_csv_with_schema(schema_path, csv_path, table_ref, client):
    """
    Reads a CSV into pandas, reorders its columns to match the JSON schema’s key order,
    then loads it into BigQuery using the given table_ref and client.
    """
    print("preparing to load CSV into BigQuery...")
    # 1. Load the schema JSON (insertion order of keys is preserved in Python 3.7+)
    with open(schema_path, "r") as f:
        schema_json = json.load(f)

    # 2. Read the CSV into a DataFrame.
    # dtype=str prevents pandas from inferring code columns (bill_type_code,
    # drg_code, admit_source_code, place_of_service_code, etc.) as float64,
    # which would strip leading zeros and append ".0" (e.g. "0111" -> "111.0").
    # BQ's CSV loader parses numeric-typed columns from string just fine, so
    # forcing str is safe across the board. keep_default_na=False keeps empty
    # cells as empty strings, which BQ loads as NULL for nullable columns.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # 3. Ensure every field in schema exists in the DataFrame (fill missing columns with None)
    for field in schema_json.keys():
        if field not in df.columns:
            df[field] = None

    # 4. Reorder DataFrame columns to exactly match the JSON key order
    df = df[list(schema_json.keys())]

    # 5. Write the reordered DataFrame back to a CSV in memory
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # 6. Convert JSON schema to BigQuery SchemaField list
    bq_schema = [
        bigquery.SchemaField(name, ftype)
        for name, ftype in schema_json.items()
    ]

    # 7. Configure and run the load job
    job_config = bigquery.LoadJobConfig(
        schema=bq_schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1  # because df.to_csv() already wrote the header row
    )

    print("Loading CSV into BigQuery...")
    load_job = client.load_table_from_file(
        csv_buffer,
        table_ref,
        job_config=job_config
    )
    load_job.result()  # wait for completion
    print(f"Loaded {csv_path} into {table_ref.path}")


def count_refs(sql, model_name):
    """Count occurrences of {{ ref('model_name') }} in the raw dbt model."""
    pattern = r"\{\{\s*ref\(['\"]%s['\"]\)\s*\}\}" % re.escape(model_name)
    return len(re.findall(pattern, sql))


def add_scaled_inplace(dst: dict, src: dict, scale: int | float = 1):
    """Elementwise: dst[k] += scale * src[k] for numeric values only."""
    if len(src) != 0:
        for k, v in src.items():
            if isinstance(v, (int, float)):
                dst[k] = dst.get(k, 0) + scale * v


def skeleton_jaccard(skel_a: dict, skel_b: dict) -> float:
    """Multiset Jaccard similarity between two skeleton token dicts."""
    all_keys = set(skel_a) | set(skel_b)
    if not all_keys:
        return 0.0
    num = sum(min(skel_a.get(k, 0), skel_b.get(k, 0)) for k in all_keys)
    den = sum(max(skel_a.get(k, 0), skel_b.get(k, 0)) for k in all_keys)
    return num / den if den > 0 else 0.0


def _compute_pair_diff(skel_a: dict, skel_b: dict,
                       struct_a: dict, struct_b: dict) -> dict:
    """Compute combined skeleton + structural feature diff between two models."""
    diff: dict = {}
    # skeleton token diff
    for k in set(skel_a) | set(skel_b):
        d = skel_a.get(k, 0) - skel_b.get(k, 0)
        if d != 0:
            diff[f"skel:{k}"] = d
    # structural feature diff
    for k in set(struct_a) | set(struct_b):
        d = struct_a.get(k, 0) - struct_b.get(k, 0)
        if d != 0:
            diff[f"struct:{k}"] = d
    return diff


def build_sibling_pairs(
    skeleton_map: dict,
    structural_map: dict,
    table_stats: dict,
    jaccard_threshold: float = 0.2,
    max_pairs_per_model: int = 100,
) -> Tuple[List[dict], np.ndarray]:
    """
    Build training pairs from sibling models with high skeleton overlap.

    Returns (X_diff_dicts, y_log_ratio) where each entry corresponds to a
    pair (Mi, Mj) and y = log1p(rows_i) - log1p(rows_j).
    """
    # anchor UIDs: present in both skeleton_map and table_stats with rows_affected
    anchors = sorted(
        uid for uid in skeleton_map
        if uid in table_stats and table_stats[uid].get("rows_affected") is not None
    )

    X_diffs: List[dict] = []
    y_ratios: List[float] = []
    pair_count: Dict[str, int] = {}

    for i in range(len(anchors)):
        uid_a = anchors[i]
        if pair_count.get(uid_a, 0) >= max_pairs_per_model:
            continue
        for j in range(i + 1, len(anchors)):
            uid_b = anchors[j]
            if pair_count.get(uid_b, 0) >= max_pairs_per_model:
                continue

            rows_a = float(table_stats[uid_a]["rows_affected"])
            rows_b = float(table_stats[uid_b]["rows_affected"])
            # skip if both empty
            if rows_a == 0 and rows_b == 0:
                continue

            skel_a = skeleton_map.get(uid_a, {})
            skel_b = skeleton_map.get(uid_b, {})
            jac = skeleton_jaccard(skel_a, skel_b)
            if jac < jaccard_threshold:
                continue

            struct_a = structural_map.get(uid_a, {})
            struct_b = structural_map.get(uid_b, {})
            diff = _compute_pair_diff(skel_a, skel_b, struct_a, struct_b)

            X_diffs.append(diff)
            y_ratios.append(log1p(rows_a) - log1p(rows_b))
            pair_count[uid_a] = pair_count.get(uid_a, 0) + 1
            pair_count[uid_b] = pair_count.get(uid_b, 0) + 1

    return X_diffs, np.asarray(y_ratios, dtype=float)


def fit_sibling_diff_model(X_diff_dicts: list, y_log_ratio: np.ndarray,
                           random_state: int = 42) -> dict:
    """Train a Huber regressor on sibling-pair diffs to predict log-cardinality ratio."""
    if len(X_diff_dicts) < 5:
        raise RuntimeError(f"Not enough sibling pairs to train (got {len(X_diff_dicts)}).")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(X_diff_dicts)

    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000),
    )
    pipe.fit(X, y_log_ratio)
    return {"dv": dv, "pipe": pipe}


def predict_cardinality_sibling(
    uid: str,
    skeleton_map: dict,
    structural_map: dict,
    table_stats: dict,
    sibling_model: dict,
    top_k: int = 5,
    min_jaccard: float = 0.2,
) -> Optional[float]:
    """
    Predict cardinality for `uid` by comparing to known sibling models.

    Returns predicted rows_affected, or None if no anchor meets min_jaccard.
    """
    skel_target = skeleton_map.get(uid, {})
    struct_target = structural_map.get(uid, {})

    # score all anchors
    scored: List[Tuple[float, str]] = []
    for anchor_uid, stats in table_stats.items():
        if anchor_uid == uid:
            continue
        if stats.get("rows_affected") is None:
            continue
        skel_anchor = skeleton_map.get(anchor_uid, {})
        jac = skeleton_jaccard(skel_target, skel_anchor)
        # print(f"Jaccard between {uid} and {anchor_uid}: {jac:.3f}")
        
        if jac >= min_jaccard:
            scored.append((jac, anchor_uid))

    if not scored:
        return None

    # top-k by Jaccard
    scored.sort(reverse=True)
    top_anchors = scored[:top_k]
    
    # print(f"size of top anchors: {len(top_anchors)}")

    dv = sibling_model["dv"]
    pipe = sibling_model["pipe"]

    estimates: List[float] = []
    for _, anchor_uid in top_anchors:
        anchor_rows = float(table_stats[anchor_uid]["rows_affected"])
        skel_anchor = skeleton_map.get(anchor_uid, {})
        struct_anchor = structural_map.get(anchor_uid, {})
        diff = _compute_pair_diff(skel_target, skel_anchor, struct_target, struct_anchor)
        log_ratio = pipe.predict(dv.transform(diff))[0]
        est = log1p(anchor_rows) + log_ratio
        estimates.append(est)

    pred = float(np.expm1(np.median(estimates)))
    
    # print(f"Predicted rows_affected for {uid}: {pred} based on {len(estimates)} siblings")
    
    return max(pred, 0.0)


def dry_run_bytes_for_sql(client: bigquery.Client, sql: str) -> int:
    """
    Returns total bytes processed for `sql` without executing it.
    If the LIMIT is present, BigQuery still reports full scan bytes.
    """
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    try:
        job = client.query(sql, job_config=job_config)
        # Note: dry run doesn't execute or bill, but populates this field.
        return int(job.total_bytes_processed or 0)
    except BadRequest as e:
        # If parsing/permissions fail, return 0 (or re-raise if you prefer)
        print(f"[dry-run failed] {e.message}")
        return -1
    






def build_dataset(feature_map: dict, target_map: dict, target_key: str):
    """Align feature dicts with targets from table_stats_dict[target_key]."""
    X_dicts, y, uids = [], [], []
    for uid, stats in target_map.items():
        feats = feature_map.get(uid)
        if not feats:  # missing or empty feature dict
            continue
        y_val = stats.get(target_key)
        if y_val is None:
            continue
        X_dicts.append(feats)
        y.append(float(y_val))
        uids.append(uid)
    return X_dicts, np.asarray(y, dtype=float), uids


class ConstrainedHuberRegressor:
    """Huber regression with sign constraints on coefficients.

    Uses scipy L-BFGS-B to minimize Huber loss with per-coefficient bounds,
    enforcing that certain features have non-negative (or non-positive) weights.
    Maintains sklearn-compatible .predict() interface.
    """

    def __init__(self, epsilon=1.35, alpha=1e-4, max_iter=2000, sign_constraints=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.sign_constraints = sign_constraints or {}  # {feature_name: +1 or -1}
        self.coef_ = None
        self.intercept_ = None

    def _huber_loss(self, params, X, y):
        w = params[:-1]
        b = params[-1]
        residuals = y - X @ w - b
        abs_r = np.abs(residuals)
        mask = abs_r <= self.epsilon
        loss = (np.sum(0.5 * residuals[mask] ** 2)
                + np.sum(self.epsilon * abs_r[~mask] - 0.5 * self.epsilon ** 2))
        reg = 0.5 * self.alpha * np.sum(w ** 2)
        return loss / len(y) + reg

    def _huber_grad(self, params, X, y):
        w = params[:-1]
        b = params[-1]
        residuals = y - X @ w - b
        abs_r = np.abs(residuals)
        mask = abs_r <= self.epsilon

        grad_r = np.zeros_like(residuals)
        grad_r[mask] = -residuals[mask]
        grad_r[~mask] = -self.epsilon * np.sign(residuals[~mask])
        grad_r /= len(y)

        grad_w = X.T @ grad_r + self.alpha * w
        grad_b = np.sum(grad_r)
        return np.append(grad_w, grad_b)

    def fit(self, X, y, feature_names=None):
        from scipy.optimize import minimize as sp_minimize

        if hasattr(X, 'toarray'):
            X = X.toarray()

        n_features = X.shape[1]
        bounds = []
        for i in range(n_features):
            fname = feature_names[i] if feature_names is not None and i < len(feature_names) else None
            sign = self.sign_constraints.get(fname)
            if sign == 1:
                bounds.append((0, None))
            elif sign == -1:
                bounds.append((None, 0))
            else:
                bounds.append((None, None))
        bounds.append((None, None))  # intercept

        x0 = np.zeros(n_features + 1)
        x0[-1] = np.mean(y)

        result = sp_minimize(
            self._huber_loss, x0, args=(X, y),
            jac=self._huber_grad,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': 1e-12},
        )

        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        return self

    def predict(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return X @ self.coef_ + self.intercept_


def fit_huber_dictreg(X_dicts, y_raw, random_state=42, sign_constraints=None):
    if len(X_dicts) < 5:
        raise RuntimeError(f"Not enough samples to train (got {len(X_dicts)}).")

    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(X_dicts)
    y_log = np.log1p(y_raw)

    if sign_constraints:
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X)
        feature_names = list(dv.get_feature_names_out())
        model = ConstrainedHuberRegressor(
            epsilon=1.35, alpha=1e-4, max_iter=2000,
            sign_constraints=sign_constraints,
        )
        model.fit(X_scaled, y_log, feature_names=feature_names)
        pipe = make_pipeline(scaler, model)
        pipe.steps[0] = ('standardscaler', scaler)
        pipe.steps[1] = ('constrainedhuberregressor', model)
    else:
        pipe = make_pipeline(
            StandardScaler(with_mean=False),
            HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)
        )
        pipe.fit(X, y_log)

    return pipe, dv


def build_dataset_two_part(scale_map, skel_map, target_map, target_key):
    Xs, Xk, y, uids = [], [], [], []
    for uid, st in target_map.items():
        if target_key not in st: 
            continue
        scale = scale_map.get(uid)
        if not scale: 
            continue
        Xs.append(scale)
        Xk.append(skel_map.get(uid, {}))
        y.append(float(st[target_key]))
        uids.append(uid)
    return Xs, Xk, np.asarray(y, float), uids


def fit_elasticnet_two_part(X_scale_dicts, X_skel_dicts, y_raw, n_hash=2**16, random_state=42):
    if len(X_scale_dicts) < 5:
        raise RuntimeError(f"Not enough samples to train (got {len(X_scale_dicts)}).")

    dv = DictVectorizer(sparse=True)
    X1 = dv.fit_transform(X_scale_dicts)

    hasher = FeatureHasher(n_features=n_hash, input_type="dict", alternate_sign=False)
    X2 = hasher.transform([d or {} for d in X_skel_dicts])

    X = sp.hstack([X1, X2], format="csr")
    y_log = np.log1p(y_raw)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("enet", ElasticNet(alpha=1e-3, l1_ratio=0.5, max_iter=30000, random_state=random_state)),
    ])
    pipe.fit(X, y_log)

    return {"dv": dv, "hasher": hasher, "pipe": pipe}


def align_by_uid(uids_a, Xa, ya, uids_b, Xb1, Xb2, yb):
    ia = {u:i for i,u in enumerate(uids_a)}
    ib = {u:i for i,u in enumerate(uids_b)}
    common = sorted(set(ia) & set(ib))

    Xa_  = [Xa[ia[u]] for u in common]
    Xb1_ = [Xb1[ib[u]] for u in common]
    Xb2_ = [Xb2[ib[u]] for u in common]
    y_   = np.asarray([ya[ia[u]] for u in common], float)
    y2_  = np.asarray([yb[ib[u]] for u in common], float)

    if not np.allclose(y_, y2_):
        raise RuntimeError("Target mismatch after UID alignment.")
    return common, Xa_, Xb1_, Xb2_, y_


# ---------- NEW: empty classifier (skeleton only) trained on ALL data ----------
def fit_empty_classifier_skeleton(X_skel_dicts, y_raw, n_hash=2**16, random_state=42):
    y_empty = (np.asarray(y_raw, float) <= 0).astype(int)

    hasher = FeatureHasher(n_features=n_hash, input_type="dict", alternate_sign=False)
    X = hasher.transform([d or {} for d in X_skel_dicts])

    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y_empty)

    # choose tau by maximizing F1 on TRAIN (in-sample)
    p = clf.predict_proba(X)[:, 1]
    prec, rec, thr = precision_recall_curve(y_empty, p)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best = int(np.nanargmax(f1))
    tau = float(thr[max(best - 1, 0)]) if len(thr) > 0 else 0.5
    yhat = (p >= tau).astype(int)
    print(f"[empty-clf] empty_rate={y_empty.mean():.3f}  train_F1={f1_score(y_empty, yhat):.3f}  tau={tau:.3f}")

    return {"hasher": hasher, "clf": clf, "tau": tau}





# slot time ILP
def solve_slot_time_ilp(
    B: Dict[str, float],                         # baseline cost per child j
    U: Dict[Tuple[str, str], float],             # edge deltas U[(i,j)]
    edges: Iterable[Tuple[str, str]],            # list/iter of (i,j)
    targets: Iterable[str],                      # nodes that must be built as tables
    t_ref: Optional[Dict[str, int]] = None,      # reference materialization {node:0/1}
    trust_region_L: Optional[int] = None,        # max #nodes allowed to flip this iter
    flip_penalty: float = 0.0,                   # optional λ * (#flips)
    solver_name: str = "highs",                  # "highs", "cbc", "glpk", "gurobi", ...
    time_limit_s: Optional[int] = None,
):
    """
    Baseline + deltas objective:
      minimize  sum_j t[j] * B[j]  +  sum_(i,j) y[i,j] * U[(i,j)]

    Variables:
      t[i] in {0,1}: 1 => materialize node i as table (executes)
      y[i,j] in [0,1] : 1 iff (i is view) AND (j executes)

    Linearization:
      y[i,j] <= 1 - t[i]
      y[i,j] <= t[j]
      y[i,j] >= t[j] - t[i]
      y[i,j] >= 0
    """
    # --- Build node set from inputs
    V = set(B.keys())
    for (i, j) in edges:
        V.add(i); V.add(j)
    V = sorted(V)
    E = list(edges)
    T = set(targets)

    # Sanity checks
    missing_B = [j for j in V if j in B and not isinstance(B[j], (int, float))]
    missing_U = [e for e in E if e not in U]
    if missing_U:
        raise ValueError(f"U is missing {len(missing_U)} edges (e.g., {missing_U[:3]})")

    model = pyo.ConcreteModel()
    model.V = pyo.Set(initialize=V, ordered=True)
    model.E = pyo.Set(initialize=E, dimen=2, ordered=False)

    # Parameters
    model.B = pyo.Param(model.V, initialize=lambda m,i: float(B.get(i, 0.0)), within=pyo.NonNegativeReals)
    model.U = pyo.Param(model.E, initialize=lambda m,i,j: float(U[(i,j)]))

    # Vars
    model.t = pyo.Var(model.V, within=pyo.Binary)      # table vs view
    model.y = pyo.Var(model.E, bounds=(0.0, 1.0))      # continuous; will end up 0/1

    # Targets must be tables
    def _target_rule(m, i):
        return m.t[i] == 1 if i in T else pyo.Constraint.Skip
    model.target_con = pyo.Constraint(model.V, rule=_target_rule)

    # Linearization: y_ij = (1 - t_i) * t_j (enforced with inequalities)
    def _y1(m, i, j): return m.y[(i,j)] <= 1 - m.t[i]
    def _y2(m, i, j): return m.y[(i,j)] <= m.t[j]
    def _y3(m, i, j): return m.y[(i,j)] >= m.t[j] - m.t[i]
    model.y1 = pyo.Constraint(model.E, rule=_y1)
    model.y2 = pyo.Constraint(model.E, rule=_y2)
    model.y3 = pyo.Constraint(model.E, rule=_y3)


    # Optional trust region around t_ref
    if t_ref is not None and (trust_region_L is not None or flip_penalty > 0.0):
        # binary flip indicator d_i >= |t_i - t_ref_i|
        model.d = pyo.Var(model.V, within=pyo.Binary)
        # d_i >= t_i - t_ref_i and d_i >= t_ref_i - t_i
        def _flip1(m, i): return m.d[i] >= m.t[i] - int(t_ref.get(i, 0))
        def _flip2(m, i): return m.d[i] >= int(t_ref.get(i, 0)) - m.t[i]
        model.flip1 = pyo.Constraint(model.V, rule=_flip1)
        model.flip2 = pyo.Constraint(model.V, rule=_flip2)
        if trust_region_L is not None:
            model.trust = pyo.Constraint(expr = sum(model.d[i] for i in model.V) <= int(trust_region_L))
    else:
        model.d = None  # not used

    # Objective
    flip_term = (flip_penalty * sum(model.d[i] for i in model.V)) if (model.d is not None and flip_penalty > 0.0) else 0.0
    model.obj = pyo.Objective(
        expr = sum(model.t[j] * model.B[j] for j in model.V) +
               sum(model.y[(i,j)] * model.U[(i,j)] for (i,j) in model.E) +
               flip_term,
        sense=pyo.minimize
    )

    # Warm start
    if t_ref is not None:
        for i in V:
            if i in t_ref:
                model.t[i].value = int(t_ref[i])

    # Solve
    opt = pyo.SolverFactory(solver_name)
    if opt is None or not opt.available():
        raise RuntimeError(f"Solver '{solver_name}' not available. Install e.g. 'highs' or 'cbc'.")
    if solver_name in ("gurobi", "gurobi_persistent") and time_limit_s:
        opt.set_options({"TimeLimit": time_limit_s})
    elif solver_name == "highs" and time_limit_s:
        opt.options["time_limit"] = time_limit_s
    elif solver_name in ("cbc","glpk") and time_limit_s:
        # Many open solvers ignore time limits; best effort:
        opt.options["seconds"] = time_limit_s

    res = opt.solve(model, tee=False)

    # Extract solution
    t_sol = {i: int(round(pyo.value(model.t[i]))) for i in V}
    y_sol = {(i,j): float(pyo.value(model.y[(i,j)])) for (i,j) in E}
    obj_val = float(pyo.value(model.obj))
    flips = None
    if model.d is not None:
        flips = {i: int(round(pyo.value(model.d[i]))) for i in V}

    return t_sol, y_sol, obj_val, flips, res.solver.termination_condition


# y_ij = (1 - t_i) * t_j
def _y(t: Dict[str, int], i: str, j: str) -> int:
    return (1 - int(t.get(i, 0))) * int(t.get(j, 0))

def evaluate_objective(B: Dict[str, float],
                       U: Dict[Tuple[str, str], float],
                       edges: Iterable[Tuple[str, str]],
                       t: Dict[str, int],
                       flip_penalty: float = 0.0,
                       t_ref: Dict[str, int] = None) -> float:
    """Objective = sum_j t_j * B_j + sum_(i,j) y_ij * U_ij + flip_penalty * #flips (optional)."""
    base = sum(float(B.get(j, 0.0)) * int(t.get(j, 0)) for j in set(B.keys()) | {j for _, j in edges})
    edge = sum(float(U[(i, j)]) * _y(t, i, j) for (i, j) in edges)
    flips = 0
    if flip_penalty and t_ref is not None:
        flips = sum(1 for n in set(t.keys()) | set(t_ref.keys()) if int(t.get(n, 0)) != int(t_ref.get(n, 0)))
    return base + edge + flip_penalty * flips

def per_child_costs(B: Dict[str, float],
                    U: Dict[Tuple[str, str], float],
                    edges: Iterable[Tuple[str, str]],
                    t: Dict[str, int]) -> Dict[str, float]:
    """Cost attributed to each child j: t_j * B_j + sum_{i -> j} y_ij * U_ij."""
    parents_of: Dict[str, List[str]] = {}
    for i, j in edges:
        parents_of.setdefault(j, []).append(i)
    cost = {}
    nodes = set(B.keys()) | set(parents_of.keys()) | {j for _, j in edges}
    for j in nodes:
        tj = int(t.get(j, 0))
        base = float(B.get(j, 0.0)) * tj
        inc = sum(float(U[(i, j)]) * _y(t, i, j) for i in parents_of.get(j, []))
        cost[j] = base + inc
    return cost

def changed_nodes(t_old: Dict[str, int], t_new: Dict[str, int]) -> List[str]:
    nodes = set(t_old.keys()) | set(t_new.keys())
    return sorted([n for n in nodes if int(t_old.get(n, 0)) != int(t_new.get(n, 0))])

def summarize_iteration(B, U, edges, t_before, t_after,
                        flip_penalty: float = 0.0, t_ref=None, top_k: int = 15) -> str:
    obj_before = evaluate_objective(B, U, edges, t_before, flip_penalty, t_ref)
    obj_after  = evaluate_objective(B, U, edges, t_after,  flip_penalty, t_ref)
    delta = obj_after - obj_before

    # Which models changed?
    flips = changed_nodes(t_before, t_after)

    # Per-child cost movement (who actually got cheaper/more expensive)
    c_before = per_child_costs(B, U, edges, t_before)
    c_after  = per_child_costs(B, U, edges, t_after)
    movers = [(j, c_before.get(j,0.0), c_after.get(j,0.0), c_after.get(j,0.0)-c_before.get(j,0.0))
              for j in sorted(set(c_before.keys()) | set(c_after.keys()))]
    movers.sort(key=lambda x: abs(x[3]), reverse=True)

    # Build a small text report
    lines = []
    lines.append(f"Objective (this iteration's B,U): BEFORE = {obj_before:,.2f}, AFTER = {obj_after:,.2f}, Δ = {delta:,.2f}")
    lines.append(f"Models flipped ({len(flips)}): {', '.join(flips) if flips else '(none)'}")
    # lines.append(f"Top {min(top_k, len(movers))} per-child cost changes (slot_ms):")
    # for j, cb, ca, d in movers[:top_k]:
    #     mark = " *" if j in flips else ""
    #     lines.append(f"  - {j:>20}{mark} : before={cb:,.2f}, after={ca:,.2f}, Δ={d:,.2f}")
    return "\n".join(lines)



def load_manifest(manifest_path: str | Path) -> dict:
    """Load dbt manifest.json from disk."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_run_results(run_results_path: str | Path) -> dict:
    """Load dbt run_results.json from disk."""
    run_results_path = Path(run_results_path)
    output = {}
    with run_results_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
        res = obj["results"]
        for r in res:
            output[r["unique_id"]] = r["adapter_response"]["slot_ms"]
    return output


def get_total_slot_time(run_results_path: str | Path):
    run_results = json.loads(Path(run_results_path).read_text())
    total_slot_ms = 0
    for result in run_results["results"]:
        uid = result["unique_id"]
        ar = result.get("adapter_response", {})
        slot_ms = ar.get("slot_ms", 0)
        total_slot_ms += slot_ms
        
    total_slot_seconds = total_slot_ms / 1000.0
    return total_slot_seconds


def build_parent_child_maps(manifest: Mapping) -> tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Return (parent_map, child_map) for all nodes/sources in the manifest.

    parent_map[node]  = set of upstream nodes
    child_map[node]   = set of downstream nodes
    """
    nodes = manifest.get("nodes", {})
    sources = manifest.get("sources", {})

    # dbt >= 0.20 usually provides these directly
    parent_map: Dict[str, List[str]] = dict(manifest.get("parent_map", {}))
    child_map: Dict[str, List[str]] = dict(manifest.get("child_map", {}))

    # If absent or incomplete, build them from node.depends_on.nodes
    if not parent_map:
        parent_map = {}
        for unique_id, node in {**nodes, **sources}.items():
            parents = node.get("depends_on", {}).get("nodes", [])
            parent_map[unique_id] = set(parents)

    if not child_map:
        child_map = {uid: set() for uid in {**nodes, **sources}}
        for child, parents in parent_map.items():
            for p in parents:
                child_map.setdefault(p, set()).add(child)

    return parent_map, child_map


def human_readable_dag_summary(
    manifest: Mapping,
    node_ids: Iterable[str],
    restrict_to_subset: bool = True,
) -> str:
    """
    Build a human-readable summary of dependencies for given dbt node IDs.

    Args:
        manifest: Parsed manifest.json as a dict.
        node_ids: Iterable of dbt unique_ids (e.g. "model.myproj.my_model").
        restrict_to_subset:
            - If False (default): show upstream/downstream including nodes
              outside the given set.
            - If True: only keep edges where both ends are in node_ids.

    Returns:
        A multi-line string suitable for feeding to an LLM.
    """
    nodes = manifest.get("nodes", {})
    sources = manifest.get("sources", {})
    parent_map, child_map = build_parent_child_maps(manifest)

    subset = set(node_ids)
    lines: List[str] = ["DBT MODELS AND DEPENDENCIES (DAG)", ""]

    for unique_id in node_ids:
        node = nodes.get(unique_id) or sources.get(unique_id)
        if node is None:
            lines.append(f"Node: {unique_id} (NOT FOUND in manifest)")
            lines.append("")
            continue

        name = node.get("name", unique_id)
        resource_type = node.get("resource_type", "unknown")

        upstream_full = parent_map.get(unique_id, [])
        downstream_full = child_map.get(unique_id, [])

        if restrict_to_subset:
            upstream = [u for u in upstream_full if u in subset]
            downstream = [d for d in downstream_full if d in subset]
        else:
            upstream = upstream_full
            downstream = downstream_full

        lines.append(
            f"Node: {unique_id} (name={name}, resource_type={resource_type})"
        )
        lines.append(f"  Upstream: {upstream if upstream else []}")
        lines.append(f"  Downstream: {downstream if downstream else []}")
        lines.append("")

    return "\n".join(lines)

# input: a list of lists of strings (subgroups)
def human_readable_subgroup_summary(
    subgroup_lists: Iterable[Iterable[str]]
) -> str:
    # we will select the first one as representative and say that others are similar
    lines: List[str] = ["DBT MODEL SUBGROUPS BASED ON SIMILARITY", ""]
    for subgroup in subgroup_lists:
        subgroup = list(subgroup)
        if not subgroup:
            continue
        representative = subgroup[0]
        others = subgroup[1:]
        lines.append(f"Subgroup representative: {representative}")
        if others:
            lines.append(f"  Similar models in this subgroup: {others}")
        else:
            lines.append(f"  (No other similar models in this subgroup)")
        lines.append("")
    return "\n".join(lines)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text."""
    # GPT-5 family tokenization is o200k_base (safe fallback for plain-text token counts)
    encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(text))


def predict_elasticnet(dv, hasher, pipe, structural_dict, skeleton_dict):
    """Predict using the two-part ElasticNet model (structural + skeleton features)."""
    X1 = dv.transform(structural_dict)
    X2 = hasher.transform([skeleton_dict or {}])
    X = sp.hstack([X1, X2], format="csr")
    return pipe.predict(X)


def compute_dag_cost(
    features_json_path: str,
    graph_json_path: str,
    materialization_json_path: str,
    trained_models_bundle_path: str,
) -> dict:
    """
    Single-pass cost evaluation of a DAG under a given materialization.

    Returns dict with keys: 'cost', 'B', 'U', 't', 'num_models', 'edges'.
    """
    import joblib
    import networkx as nx
    from networkx.readwrite import json_graph
    from collections import defaultdict
    from copy import deepcopy

    # --- load features ---
    with open(features_json_path) as f:
        features = json.load(f)
    time_features = features.get("time_features_per_model", {})
    cardinality_features = features.get("cardinality_features_per_model", {})
    all_table_stats = features.get("all_table_stats", {})

    # --- load graph ---
    with open(graph_json_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data)
    topo = list(nx.topological_sort(G))
    targets = [n for n in G if G.out_degree(n) == 0 and G.in_degree(n) > 0]
    edges = list(G.edges())

    # --- load converged materialization ---
    with open(materialization_json_path) as f:
        materialization = json.load(f)

    # --- load trained models ---
    with open(trained_models_bundle_path, "rb") as f:
        bundle = joblib.load(f)
    card_dv = bundle["cardinality"]["huber_all"]["dv"]
    card_pipe = bundle["cardinality"]["huber_all"]["pipe"]
    time_dv = bundle["slot_time"]["huber"]["dv"]
    time_pipe = bundle["slot_time"]["huber"]["pipe"]

    # --- feature propagation under given materialization ---
    current_table_cardinality = {
        k: v.get("rows_affected", 0)
        for k, v in all_table_stats.items()
    }

    # First pass: expand features and propagate cardinality
    sum_parents_cardinality_dict = defaultdict(int)
    expanded_time = deepcopy(time_features)
    expanded_cardinality = deepcopy(cardinality_features)
    for u in topo:
        if u in materialization and materialization.get(u) not in ["table"]:
            for v in G.successors(u):
                w = G[u][v].get("weight", 1)
                add_scaled_inplace(expanded_time[v], expanded_time[u], w)
                add_scaled_inplace(expanded_cardinality[v], expanded_cardinality[u], w)

        if materialization.get(u) in ["table"] or u not in materialization:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += current_table_cardinality.get(u, 0) * G[u][v].get("weight", 1)
        else:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += sum_parents_cardinality_dict.get(u, 0) * G[u][v].get("weight", 1)

    for k in expanded_cardinality:
        expanded_cardinality[k]["sum_parents_cardinality_log1p"] = np.log1p(
            sum_parents_cardinality_dict.get(k, 0)
        )

    # --- predict cardinality ---
    current_predicted_cardinality = {
        k: v.get("rows_affected", 0) for k, v in all_table_stats.items()
    }
    for k in materialization:
        if k not in all_table_stats:
            pred_log = card_pipe.predict(card_dv.transform(expanded_cardinality[k]))
            current_predicted_cardinality[k] = max(0.0, np.expm1(pred_log)[0])

    # Include predicted cardinalities for table-materialized models in
    # current_table_cardinality, matching the ILP's behavior.
    # Without this, models flipped from view->table by the ILP contribute
    # cardinality=0 to sum_parents_cardinality_dict, diverging from the ILP.
    current_table_cardinality.update({
        k: current_predicted_cardinality[k]
        for k in materialization
        if materialization[k] == "table" and k not in all_table_stats
    })

    # Second pass: re-propagate sum_parents_cardinality_dict with updated
    # current_table_cardinality (now includes predicted cardinalities for
    # table models that lack actual stats).
    sum_parents_cardinality_dict = defaultdict(int)
    for u in topo:
        if materialization.get(u) in ["table"] or u not in materialization:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += current_table_cardinality.get(u, 0) * G[u][v].get("weight", 1)
        else:
            for v in G.successors(u):
                sum_parents_cardinality_dict[v] += sum_parents_cardinality_dict.get(u, 0) * G[u][v].get("weight", 1)

    # --- predict B (per-node baseline cost) ---
    # Clamp to prevent numerical overflow from out-of-distribution feature propagation.
    B_MAX = np.expm1(20.0)  # ~485M ms — well above any real BigQuery query

    B = {}
    for k in materialization:
        if k not in G:
            continue
        this_time_feature = time_features[k].copy()
        sum_parents_card = sum(
            current_predicted_cardinality.get(p, 0) for p in G.predecessors(k)
        )
        this_time_feature["sum_parents_cardinality_log1p"] = np.log1p(sum_parents_card)
        this_time = np.expm1(
            time_pipe.predict(time_dv.transform(this_time_feature))
        )[0]
        B[k] = min(max(0.0, this_time), B_MAX)

    # --- predict U (edge deltas) ---
    U = {}
    for k in materialization:
        if k not in G:
            continue
        for p in G.predecessors(k):
            this_time_feature = time_features[k].copy()
            if p in expanded_time:
                w = G[p][k].get("weight", 1)
                add_scaled_inplace(this_time_feature, expanded_time[p], w)

            this_sum_parents_card = 0
            for pp in G.predecessors(k):
                if pp == p:
                    this_sum_parents_card += sum_parents_cardinality_dict.get(pp, 0)
                else:
                    this_sum_parents_card += current_predicted_cardinality.get(pp, 0)

            this_time_feature["sum_parents_cardinality_log1p"] = np.log1p(this_sum_parents_card)
            this_time = np.expm1(
                time_pipe.predict(time_dv.transform(this_time_feature))
            )[0]
            this_time = min(max(0.0, this_time), B_MAX)
            U[(p, k)] = max(min(this_time - B[k], B_MAX), -B_MAX)

    # --- build t and evaluate ---
    t = {}
    for k, v in materialization.items():
        if k in G:
            t[k] = 1 if v == "table" else 0

    cost = evaluate_objective(B, U, edges, t)

    return {
        "cost": cost,
        "B": B,
        "U": U,
        "t": t,
        "num_models": len(materialization),
        "edges": edges,
    }