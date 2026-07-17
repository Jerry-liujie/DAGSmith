# dfexpr_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import hashlib, json, math, re


from .dfexpr import DFExpr, DFRef, DFLiteral, DFTransform, ColumnDF, PredicateDF, JoinDF

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class DFHashConfig:
    """Controls how we canonicalize & hash DFExpr trees."""
    anonymize_literals: bool = False           # replace literals with typed placeholders
    literal_string_maxlen: int = 16           # when anonymize=False, clamp long strings (for JSON)
    float_round: int = 6                      # round floats to this many decimals
    include_relation_in_ref: bool = True      # prefer relation-qualified refs in fingerprints
    include_model_uid_in_ref: bool = True     # fall back to model UID if relation is not available
    include_scope_id_in_ref: bool = False     # usually keep False (scope_ids vary run-to-run)
    normalize_sql_whitespace: bool = True     # collapse whitespace in sql-ish attrs (window/spec/unknown)
    sort_membership_list: bool = True         # canonicalize IN(...) options order
    sort_commutative_ops: bool = True         # canonicalize arg order for commutative ops
    flatten_associative_ops: bool = True      # flatten AND/OR/ADD/MUL trees

# =============================================================================
# Public API
# =============================================================================

def serialize_dfexpr(expr: DFExpr, *, pretty: bool = False) -> str:
    """Human/debug-friendly JSON serializer (NOT canonical)."""
    obj = _to_serializable(expr)
    return json.dumps(obj, ensure_ascii=False, indent=2 if pretty else None)

def dfexpr_to_dict(expr: DFExpr) -> Dict[str, Any]:
    """JSON-friendly (non-canonical) Python dict."""
    return _to_serializable(expr)

def fingerprint_dfexpr(expr: DFExpr, cfg: DFHashConfig = DFHashConfig()) -> str:
    """Stable SHA1 hash of a DFExpr's canonical form."""
    canon = _to_canonical(expr, cfg)
    s = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def fingerprint_predicate(pred: PredicateDF, cfg: DFHashConfig = DFHashConfig()) -> str:
    """Fingerprint a WHERE/HAVING/QUALIFY predicate (tag included)."""
    payload = {
        "kind": "predicate",
        "tag": pred.tag.upper(),
        "expr": _to_canonical(pred.expr, cfg),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def fingerprint_join(j: JoinDF, cfg: DFHashConfig = DFHashConfig()) -> str:
    """Fingerprint a JOIN ON expression + join type."""
    payload = {
        "kind": "join",
        "jtype": j.jtype.upper(),
        "expr": _to_canonical(j.expr, cfg),
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def structural_eq(a: DFExpr, b: DFExpr, cfg: DFHashConfig = DFHashConfig()) -> bool:
    """True iff two expressions are identical up to canonicalization rules in cfg."""
    return fingerprint_dfexpr(a, cfg) == fingerprint_dfexpr(b, cfg)

def iter_refs(expr: DFExpr) -> Iterable[Tuple[Optional[str], Optional[str], str]]:
    """Yield (relation, model_uid, column) for each DFRef leaf."""
    for node in _walk(expr):
        if isinstance(node, DFRef):
            yield (getattr(node, "relation", None), getattr(node, "model_uid", None), node.column)

def count_ops(expr: DFExpr) -> Dict[str, int]:
    """Return a simple op histogram for DFTransform nodes."""
    out: Dict[str, int] = {}
    for node in _walk(expr):
        if isinstance(node, DFTransform):
            out[node.op] = out.get(node.op, 0) + 1
    return out

# =============================================================================
# Internals: serialization & canonicalization
# =============================================================================

def _walk(expr: DFExpr) -> Iterable[DFExpr]:
    """Preorder traversal."""
    stack = [expr]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, DFTransform):
            # CASE keeps branches in attrs
            if n.op == "case":
                for cond, val in (n.attrs.get("branches") or []):
                    stack.append(val)
                    stack.append(cond)
                else_expr = n.attrs.get("else")
                if else_expr is not None:
                    stack.append(else_expr)
            else:
                for a in reversed(n.args):
                    stack.append(a)

def _normalize_attrs_for_json(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert attrs so everything is JSON-serializable and DFExprs are expanded."""
    def conv(v):
        if isinstance(v, (DFTransform, DFRef, DFLiteral)):
            return _to_serializable(v)
        if isinstance(v, dict):
            return {k: conv(v2) for k, v2 in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [conv(x) for x in v]
        # Leave primitives as-is
        return v
    return {k: conv(v) for k, v in (attrs or {}).items()}

def _to_serializable(expr: DFExpr) -> Dict[str, Any]:
    if isinstance(expr, DFRef):
        return {
            "type": "ref",
            "relation": getattr(expr, "relation", None),
            "model_uid": getattr(expr, "model_uid", None),
            "column": expr.column,
            "scope_id": getattr(expr, "scope_id", None),
        }

    if isinstance(expr, DFLiteral):
        return {
            "type": "lit",
            "value": expr.value,
            "dtype": getattr(expr, "dtype", None),
        }

    if isinstance(expr, DFTransform):
        # ---- CASE ----
        if expr.op == "case":
            branches = []
            for cond, val in (expr.attrs.get("branches") or []):
                branches.append([_to_serializable(cond), _to_serializable(val)])
            return {
                "type": "xform",
                "op": "case",
                "branches": branches,
                "else": (_to_serializable(expr.attrs.get("else"))
                         if expr.attrs.get("else") is not None else None),
            }

        # ---- WINDOW ----
        if expr.op == "window":
            part = [_to_serializable(p) for p in (expr.attrs.get("partition_by") or [])]

            order_out = []
            for o in (expr.attrs.get("order_by") or []):
                order_out.append({
                    "expr": _to_serializable(o.get("expr")),
                    "desc": bool(o.get("desc")),
                    "nulls_first": bool(o.get("nulls_first")),
                    "nulls_last": bool(o.get("nulls_last")),
                })

            # frame is a plain dict (strings/numbers/bools or None); normalize to JSON-friendly
            frame = _normalize_attrs_for_json(expr.attrs.get("frame"))
            # (optional) carry spec_sql if present in attrs for some dialects
            spec_sql = expr.attrs.get("spec_sql")

            payload = {
                "type": "xform",
                "op": "window",
                "args": [_to_serializable(a) for a in expr.args],  # usually [func(...)]
                "attrs": {
                    "partition_by": part,
                    "order_by": order_out,
                    "frame": frame,
                },
            }
            if spec_sql is not None:
                payload["attrs"]["spec_sql"] = spec_sql
            return payload

        # ---- SET-OP ----
        if expr.op == "set_op":
            return {
                "type": "xform",
                "op": "set_op",
                "args": [_to_serializable(a) for a in expr.args],
                "attrs": {
                    "op": expr.attrs.get("op"),
                    "distinct": bool(expr.attrs.get("distinct")),
                },
            }

        # ---- CAST ----
        if expr.op == "cast":
            return {
                "type": "xform",
                "op": "cast",
                "args": [_to_serializable(a) for a in expr.args],  # single child
                "attrs": {
                    "to_type": expr.attrs.get("to_type"),
                },
            }

        # ---- CONCAT ----
        if expr.op == "concat":
            return {
                "type": "xform",
                "op": "concat",
                "args": [_to_serializable(a) for a in expr.args],  # ordered
                "attrs": {
                    "safe": bool(expr.attrs.get("safe")),
                    "coalesce": bool(expr.attrs.get("coalesce")),
                },
            }

        # ---- AGGREGATION ----
        if expr.op == "agg":
            out = {
                "type": "xform",
                "op": "agg",
                "args": [_to_serializable(a) for a in expr.args],  # inputs to the aggregate
                "attrs": {
                    "func": (expr.attrs.get("func") or "").lower(),
                    "distinct": bool(expr.attrs.get("distinct", False)),
                    # group_keys: list of DFExpr
                    "group_keys": [_to_serializable(g) for g in (expr.attrs.get("group_keys") or [])],
                },
            }
            # ORDER BY inside aggregate (order-sensitive)
            if expr.attrs.get("order_by"):
                out["attrs"]["order_by"] = [_to_serializable(o) for o in expr.attrs["order_by"]]
            # FILTER (WHERE ...)
            if expr.attrs.get("filter") is not None:
                out["attrs"]["filter"] = _to_serializable(expr.attrs["filter"])
            return out

        # ---- GROUP KEY WRAPPER ----
        if expr.op == "group_key":
            payload = {
                "type": "xform",
                "op": "group_key",
                "args": [_to_serializable(a) for a in expr.args],  # one child
                "attrs": {},
            }
            # keep index for debugging if present
            if "index" in (expr.attrs or {}):
                payload["attrs"]["index"] = expr.attrs["index"]
            return payload

        # ---- Generic transform (func/coalesce/boolean/arithmetic/etc.) ----
        return {
            "type": "xform",
            "op": expr.op,
            "args": [_to_serializable(a) for a in expr.args],
            "attrs": _normalize_attrs_for_json(expr.attrs),
        }

    # Fallback: unknown node
    return {"type": "unknown"}


# ---------------- Canonicalization ----------------

_COMMUTATIVE = {"add", "mul", "and", "or", "eq", "ne"}
_ASSOCIATIVE = {"add", "mul", "and", "or"}

def _to_canonical(expr: DFExpr, cfg: DFHashConfig) -> Dict[str, Any]:
    """Return a canonical dict form for hashing."""
    if isinstance(expr, DFRef):
        key = _canonical_ref(expr, cfg)
        return {"t": "r", "k": key}

    if isinstance(expr, DFLiteral):
        kind, val = _canonical_literal(expr, cfg)
        return {"t": "l", "k": kind, "v": val}

    if isinstance(expr, DFTransform):

        # ---- CAST ----
        if expr.op == "cast":
            to_sql = (expr.attrs.get("to_type") or "").strip()
            if cfg.normalize_sql_whitespace:
                to_sql = _norm_sql(to_sql) or ""
            to_sql = to_sql.upper()
            child = _to_canonical(expr.args[0], cfg) if expr.args else {"t": "id"}
            return {"t": "x", "op": "cast", "to": to_sql, "a": [child]}

        # ---- CONCAT (ordered, not commutative) ----
        if expr.op == "concat":
            args = [_to_canonical(a, cfg) for a in expr.args]
            return {
                "t": "x",
                "op": "concat",
                "a": args,
                "attrs": {
                    "safe": bool(expr.attrs.get("safe")),
                    "coalesce": bool(expr.attrs.get("coalesce")),
                },
            }

        # ---- AGGREGATION ----
        if expr.op == "agg":
            func = str(expr.attrs.get("func", "")).lower()
            distinct = bool(expr.attrs.get("distinct", False))
            args = [_to_canonical(a, cfg) for a in expr.args]

            # group_keys: canonicalize & sort (order-insensitive)
            gks = expr.attrs.get("group_keys") or []
            gk_repr = [_to_canonical(g, cfg) for g in gks]
            gk_repr = sorted(gk_repr, key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":")))

            out = {"t": "x", "op": "agg", "f": func, "d": distinct, "a": args, "gk": gk_repr}

            # ORDER BY inside agg (order-sensitive)
            if expr.attrs.get("order_by"):
                ob = [_to_canonical(o, cfg) for o in expr.attrs["order_by"]]
                out["ob"] = ob

            # FILTER (WHERE ...)
            if expr.attrs.get("filter") is not None:
                out["flt"] = _to_canonical(expr.attrs["filter"], cfg)

            return out

        # ---- GROUP KEY WRAPPER ----
        if expr.op == "group_key":
            child = _to_canonical(expr.args[0], cfg) if expr.args else {"t": "id"}
            return {"t": "x", "op": "group_key", "a": [child]}

        # ---- WINDOW ----
        if expr.op == "window":
            # child function/expr
            child = _to_canonical(expr.args[0], cfg) if expr.args else {"t": "id"}

            # partition_by: order matters
            p = [ _to_canonical(x, cfg) for x in (expr.attrs.get("partition_by") or []) ]

            # order_by: each item is {"expr": DFExpr, "desc":bool, "nulls_first":bool, "nulls_last":bool}
            ob_canon = []
            for it in (expr.attrs.get("order_by") or []):
                ob_canon.append({
                    "e": _to_canonical(it.get("expr"), cfg),
                    "d": bool(it.get("desc")),
                    "nf": bool(it.get("nulls_first")),
                    "nl": bool(it.get("nulls_last")),
                })

            # frame/spec: strings normalized if configured
            fr = expr.attrs.get("frame")
            if isinstance(fr, dict):
                fr_norm = dict(fr)
                for k in ("kind", "start", "end", "sql"):
                    if isinstance(fr_norm.get(k), str) and cfg.normalize_sql_whitespace:
                        fr_norm[k] = _norm_sql(fr_norm[k])
            else:
                fr_norm = fr

            return {"t": "x", "op": "window", "a": [child], "p": p, "ob": ob_canon, "fr": fr_norm}

        # ---- CASE ----
        if expr.op == "case":
            branches = []
            for cond, val in (expr.attrs.get("branches") or []):
                branches.append([_to_canonical(cond, cfg), _to_canonical(val, cfg)])
            else_part = _to_canonical(expr.attrs.get("else"), cfg) if expr.attrs.get("else") is not None else None
            return {"t": "x", "op": "case", "b": branches, "e": else_part}

        # ---- SET OP ----
        if expr.op == "set_op":
            op = (expr.attrs.get("op") or "").upper()
            distinct = bool(expr.attrs.get("distinct"))
            args = [_to_canonical(a, cfg) for a in expr.args]
            return {"t": "x", "op": f"set::{op}", "d": distinct, "a": args}

        # ---- FUNC (generic named function) ----
        if expr.op == "func":
            name = (expr.attrs.get("name") or "").lower()
            args = [_to_canonical(a, cfg) for a in expr.args]
            return {"t": "x", "op": f"func::{name}", "a": args}

        # ---- COALESCE ----
        if expr.op == "coalesce":
            args = [_to_canonical(a, cfg) for a in expr.args]  # order matters
            return {"t": "x", "op": "coalesce", "a": args}

        # ---- Generic boolean/comparison/arithmetic & others ----
        op = expr.op.lower()
        args = [_to_canonical(a, cfg) for a in expr.args]

        # Flatten associative chains (AND/OR/ADD/MUL)
        if cfg.flatten_associative_ops and op in _ASSOCIATIVE:
            flat: List[Dict[str, Any]] = []
            for a in args:
                if a.get("t") == "x" and a.get("op") == op and "a" in a:
                    flat.extend(a["a"])
                else:
                    flat.append(a)
            args = flat

        # Sort commutative ops for stable order (and/or/add/mul/eq/ne)
        if cfg.sort_commutative_ops and op in _COMMUTATIVE:
            args = sorted(args, key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":")))

        # Membership IN(left, options...)
        if op == "in" and cfg.sort_membership_list and len(args) >= 2:
            left, opts = args[0], args[1:]
            opts = sorted(opts, key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":")))
            args = [left] + opts

        # Unknown: normalize attrs.sql if present
        if op == "unknown":
            attrs = dict(expr.attrs or {})
            sql = attrs.get("sql")
            if isinstance(sql, str) and cfg.normalize_sql_whitespace:
                attrs["sql"] = _norm_sql(sql)
            return {"t": "x", "op": op, "a": args, "attrs": attrs}

        return {"t": "x", "op": op, "a": args}

    # Fallback
    return {"t": "u"}



def _canonical_ref(ref: DFRef, cfg: DFHashConfig) -> str:
    # Build a stable key for a column reference.
    pieces: List[str] = []

    rel = getattr(ref, "relation", None)
    uid = getattr(ref, "model_uid", None)
    col = ref.column.lower() if ref.column else ref.column

    if cfg.include_relation_in_ref and rel:
        try:
            from .utils import sanitize_relation as _sr  # local import avoids hard coupling
            rel = _sr(rel).lower()
        except Exception:
            rel = rel.lower()
        pieces.append(f"rel:{rel}")
    elif cfg.include_model_uid_in_ref and uid:
        pieces.append(f"uid:{uid}")
    else:
        pieces.append("ref:?")  # unresolved leaf

    pieces.append(f"col:{col}")

    if cfg.include_scope_id_in_ref:
        sid = getattr(ref, "scope_id", None)
        if sid is not None:
            pieces.append(f"sid:{sid}")

    return "|".join(pieces)

def _canonical_literal(lit: DFLiteral, cfg: DFHashConfig) -> Tuple[str, Any]:
    # Return (kind, value) where kind in {"s","i","f","n","o"}
    if lit.value is None:
        return ("n", None)

    # Numbers
    if isinstance(lit.value, bool):
        return ("b", bool(lit.value))
    if isinstance(lit.value, int) and not isinstance(lit.value, bool):
        return ("i", int(lit.value) if not cfg.anonymize_literals else 0)
    if isinstance(lit.value, float):
        v = float(lit.value)
        if cfg.anonymize_literals:
            return ("f", 0.0)
        return ("f", round(v, cfg.float_round))

    # Strings
    if isinstance(lit.value, str):
        if cfg.anonymize_literals:
            # Keep only shape; optionally length bucket
            L = len(lit.value)
            bucket = 1 if L <= 1 else 4 if L <= 4 else 8 if L <= 8 else 16 if L <= 16 else 32 if L <= 32 else 64
            return ("s", f"<str:{bucket}>")
        # Not anonymized: clamp long strings to avoid huge JSON
        v = lit.value
        if len(v) > cfg.literal_string_maxlen:
            v = v[:cfg.literal_string_maxlen] + "…"
        return ("s", v)

    # Generic objects: stringify
    return ("o", str(lit.value))

_ws = re.compile(r"\s+")
def _norm_sql(s: Optional[str]) -> Optional[str]:
    if not s:
        return s
    s = s.strip().lower()
    s = _ws.sub(" ", s)
    return s


