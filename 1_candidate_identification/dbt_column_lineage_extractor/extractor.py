from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Iterable, Union

from time import perf_counter
import heapq
import json
import os
import sqlglot
import sqlglot.expressions as exp
from collections import defaultdict, Counter
from itertools import combinations

from .utils import *  # expects: ident_str, table_relation_str, fingerprint_expr
from .dbt_info import DbtArtifacts, DbtNode
from .dfexpr_utils import *  # expects: DFExpr, DFRef, DFLiteral, DFTransform, ColumnDF, PredicateDF, JoinDF, serialize_dfexpr, dfexpr_to_dict, fingerprint_dfexpr
from .dfexpr import *  # expects: DFExpr, DFRef, DFLiteral, DFTransform, ColumnDF, PredicateDF, JoinDF



# ============================================================
# Scope builder (Select + CTE-aware)
# ============================================================

@dataclass
class JoinInfo:
    jtype: str
    on_expr: exp.Expression  # keep original for DFExpr conversion
    source_alias: Optional[str] = None  # alias of the joined source in scope.sources




@dataclass
class Scope:
    projections: Dict[str, exp.Expression]
    proj_order: List[str]
    filters: List[Tuple[str, exp.Expression]]  # (tag, predicate)
    joins: List[JoinInfo]
    group_by_exprs: List[exp.Expression]
    sources: Dict[str, Tuple[str, Any]]        # alias -> ("table", relation_sql) | ("subquery", Scope) | ("cte", Scope)\
    has_star: bool
    scope_name: str = "scope"
    scope_id: int = -1
    # Set operation metadata (for outermost UNION/INTERSECT/EXCEPT)
    set_op: Optional[str] = None               # "UNION"|"INTERSECT"|"EXCEPT"
    set_distinct: Optional[bool] = None        # True for DISTINCT (UNION default), False for ALL
    set_children: List["Scope"] = field(default_factory=list)

@dataclass
class PruneJoinOpportunity:
    """A join that may be removable."""
    model_uid: str
    scope_name: str
    join_type: str
    unused_source_alias: str
    unused_source_ref: str            # relation string or child scope_name
    unused_source_kind: str           # "table" | "subquery" | "cte"
    confidence: str                   # "likely_safe" | "needs_verification"
    reason: str
    unused_column_count: Optional[int]
    join_on_columns_from_unused: Set[str]
    substitutions_needed: Dict[str, str]  # {B.y: A.x} — columns that must be replaced if join removed

@dataclass
class UnusedProjectionOpportunity:
    """A column projected by a model but never used downstream."""
    model_uid: str
    scope_name: str
    column_name: str
    total_projected: int

@dataclass
class PruneAnalysisResult:
    """Complete result of prune analysis."""
    prunable_joins: List[PruneJoinOpportunity]
    unused_projections: List[UnusedProjectionOpportunity]
    models_analyzed: int
    scopes_analyzed: int


class ScopeBuilder:
    def __init__(self, dialect: str):
        self.dialect = dialect
        self._next_scope_id = 1

    def _new_scope_id(self) -> int:
        sid = self._next_scope_id
        self._next_scope_id += 1
        return sid

    def _alias_of(self, node: exp.Expression) -> Optional[str]:
        alias = node.args.get("alias")
        if alias and isinstance(alias, exp.TableAlias):
            ident = alias.args.get("this")
            if isinstance(ident, exp.Identifier):
                return ident.name
        return None

    def build_scope(
        self,
        select: exp.Select,
        name: str = "scope",
        cte_map: Optional[Dict[str, "Scope"]] = None
    ) -> "Scope":
        
        # print(f"Building scope: {name}")
        
        local_cte_map: Dict[str, Scope] = dict(cte_map or {})

        # --- WITH ---
        w = select.args.get("with")
        if isinstance(w, exp.With):
            for cte in (w.args.get("expressions") or []):
                alias = None
                ta = cte.args.get("alias")
                if ta and isinstance(ta, exp.TableAlias):
                    ident = ta.args.get("this")
                    if isinstance(ident, exp.Identifier):
                        alias = ident.name
                inner = cte.args.get("this")
                if isinstance(inner, exp.Select):
                    local_cte_map[alias] = self.build_scope(inner, name=f"{name}::cte:{alias}", cte_map=local_cte_map)
                elif isinstance(inner, (exp.Union, exp.Except, exp.Intersect)):
                    # print(f"Building set-op CTE: {alias}, model name: {name}")
                    local_cte_map[alias] = self.build_set_scope(inner, name=f"{name}::cte:{alias}", cte_map=local_cte_map)
                else:
                    print(f"Warning: CTE {alias} is not a Select or set-op")
                    continue

        # --- Projections ---
        projections: Dict[str, exp.Expression] = {}
        proj_order: List[str] = []
        has_star = False
        for proj in (select.args.get("expressions") or []):
            if proj.is_star:
                has_star = True
                continue
            out_name = proj.output_name
            expr = proj.args.get("this") if isinstance(proj, exp.Alias) else proj
            projections[out_name] = expr
            proj_order.append(out_name)

        # --- Filters ---
        filters: List[Tuple[str, exp.Expression]] = []
        where = select.args.get("where")
        if isinstance(where, exp.Where):
            filters.append(("WHERE", where.args.get("this")))
        qualify = select.args.get("qualify")
        if isinstance(qualify, exp.Qualify):
            filters.append(("QUALIFY", qualify.args.get("this")))
        having = select.args.get("having")
        if isinstance(having, exp.Having):
            filters.append(("HAVING", having.args.get("this")))

        # --- Sources ---
        sources: Dict[str, Tuple[str, Any]] = {}
        def _add_source_from_table(t: exp.Table, default_label: str):
            alias = self._alias_of(t) or default_label
            base_name = ident_str(t.args.get("this"))
            if base_name in local_cte_map:
                # print(f"Found CTE source: {base_name} as {alias}")
                sources[alias] = ("cte", local_cte_map[base_name])
            else:
                # print(f"Found table source: {table_relation_str(t)} as {alias}")
                sources[alias] = ("table", table_relation_str(t))

        frm = select.args.get("from")
        if isinstance(frm, exp.From):
            t = frm.args.get("this")
            # based on my observation, for tuva project, only table option is used in FROM clause
            # the logic in isinstance(t, exp.Subquery) is not actually executed
            if isinstance(t, exp.Subquery):
                alias = self._alias_of(t) or "derived"
                inner = t.args.get("this")
                if isinstance(inner, exp.Select):
                    inner_scope = self.build_scope(inner, name=f"{name}::sub:{alias}", cte_map=local_cte_map)
                    sources[alias] = ("subquery", inner_scope)
                elif isinstance(inner, (exp.Union, exp.Except, exp.Intersect)):
                    inner_scope = self.build_set_scope(inner, name=f"{name}::sub:{alias}", cte_map=local_cte_map)
                    sources[alias] = ("subquery", inner_scope)
            elif isinstance(t, exp.Table):
                _add_source_from_table(t, t.name)


        joins: List[JoinInfo] = []
        for j in (select.args.get("joins") or []):
            jtype = (j.args.get("kind") or "INNER").upper()
            if "side" in j.args:
                jtype = j.args.get("side").upper()
            on_expr = j.args.get("on")

            ji = JoinInfo(jtype=jtype, on_expr=on_expr)
            joins.append(ji)
            t = j.args.get("this")
            if isinstance(t, exp.Subquery):
                alias = self._alias_of(t) or "derived"
                ji.source_alias = alias
                inner = t.args.get("this")
                if isinstance(inner, exp.Select):
                    inner_scope = self.build_scope(inner, name=f"{name}::join:{alias}", cte_map=local_cte_map)
                    sources[alias] = ("subquery", inner_scope)
                elif isinstance(inner, (exp.Union, exp.Except, exp.Intersect)):
                    inner_scope = self.build_set_scope(inner, name=f"{name}::join:{alias}", cte_map=local_cte_map)
                    sources[alias] = ("subquery", inner_scope)
            elif isinstance(t, exp.Table):
                ji.source_alias = self._alias_of(t) or t.name
                _add_source_from_table(t, t.name)
                
        # --- GROUP BY ---
        group_by_exprs: List[exp.Expression] = []
        grp = select.args.get("group")
        if isinstance(grp, exp.Group):
            for g in (grp.args.get("expressions") or []):
                # basic GROUP BY <expr> support
                # (ROLLUP/CUBE/GROUPING SETS can be added later if needed)
                group_by_exprs.append(g)

        # --- DISTINCT implies GROUP BY on all projected expressions ---
        distinct_str = "DISTINCT" if select.args.get("distinct") else ""
        if distinct_str:
            # it is equivalent to a group by
            for proj in (select.args.get("expressions") or []):
                if proj.is_star:
                    print(f"Warning: DISTINCT with * projection is not fully supported for backbone extraction. Scope: {name}")
                    continue
                expr = proj.args.get("this") if isinstance(proj, exp.Alias) else proj
                    
                if isinstance(expr, exp.Literal):
                    # print(f"Warning: skipping quoted identifier in DISTINCT group-by for scope: {name}, expr: {expr}")
                    continue
                group_by_exprs.append(expr)

        return Scope(
            projections=projections,
            proj_order=proj_order,
            filters=filters,
            joins=joins,
            group_by_exprs=group_by_exprs,
            sources=sources,
            has_star=has_star,
            scope_name=name,
            scope_id=self._new_scope_id(),
        )

    def build_set_scope(self, node: exp.Expression, name: str, cte_map: Dict[str, "Scope"]) -> "Scope":
        if isinstance(node, exp.Union):
            op = "UNION"
            distinct = node.args.get("distinct")
            distinct = True if distinct is None else bool(distinct)
        elif isinstance(node, exp.Intersect):
            op = "INTERSECT"
            distinct = True if node.args.get("distinct") is None else bool(node.args.get("distinct"))
        elif isinstance(node, exp.Except):
            op = "EXCEPT"
            distinct = True if node.args.get("distinct") is None else bool(node.args.get("distinct"))
        else:
            raise ValueError("Unsupported set operation")

        children: List[Scope] = []
        for child in (node.args.get("this"), node.args.get("expression")):
            if isinstance(child, exp.Subquery):
                child = child.args.get("this")
            if isinstance(child, exp.Select):
                children.append(self.build_scope(child, name=f"{name}::{op.lower()}_child{len(children)}", cte_map=cte_map))
            elif isinstance(child, (exp.Union, exp.Except, exp.Intersect)):
                children.append(self.build_set_scope(child, name=f"{name}::{op.lower()}_child{len(children)}", cte_map=cte_map))
            else:
                sel = child.find(exp.Select) if isinstance(child, exp.Expression) else None
                if sel is not None:
                    children.append(self.build_scope(sel, name=f"{name}::{op.lower()}_child{len(children)}", cte_map=cte_map))
                else:
                    raise ValueError("Set-op child is not selectable")

        projections: Dict[str, exp.Expression] = {}
        proj_order: List[str] = []
        left = children[0]
        for nm in left.proj_order:
            projections[nm] = exp.to_identifier(nm)
            proj_order.append(nm)

        return Scope(
            projections=projections,
            proj_order=proj_order,
            filters=[],
            joins=[],
            group_by_exprs=[],
            sources={},
            has_star=False,
            scope_name=name,
            scope_id=self._new_scope_id(),
            set_op=op,
            set_distinct=distinct,
            set_children=children,
        )

# ============================================================
# Extractor (expression DF + scoped predicates/joins)
# ============================================================

class Extractor:
    """
    Produces:
      - Column dataflows as DFExpr trees (pure transforms, leaves are DFRef/DFLiteral)
      - Predicate dataflows for WHERE/HAVING/QUALIFY
      - Join dataflows for ON conditions
    """

    def __init__(self, manifest: str, catalog: Optional[str] = None, dialect: str = "snowflake"):
        self.art = DbtArtifacts(manifest, catalog_path=catalog)
        self.art.load()
        self.dialect = dialect
        self.scope_builder = ScopeBuilder(dialect)
        
        # uid -> Scope
        self._uid_scope_cache: Dict[str, Scope] = {}
        
        # tuple keying: (uid, scope_id, scope_name) -> Scope
        self._full_scope_cache: Dict[Tuple[str, int, str], Scope] = {}
        
        # Memo: (uid, scope_id, projection_name) -> DFExpr
        self._memo_col: Dict[Tuple[str, int, str], DFExpr] = {}
        
        # # (uid, scope_id) -> dict of local features fingerprint
        self._scope_local_fingerprint_cache = {}
        
    # ---------------- Catalog helpers ----------------

    def relation_has_col(self, relation_sql: str, col: str) -> bool:
        key = sanitize_relation(relation_sql)
        cols = self.art.relation_columns.get(key)
        if cols and col.lower() in cols:
            return True
        # try lowercase key
        lckey = key.lower()
        cols2 = self.art.relation_columns_lc.get(lckey)
        return bool(cols2 and col.lower() in cols2)

    def column_type(self, relation_sql: str, col: str) -> Optional[str]:
        key = sanitize_relation(relation_sql)
        typemap = self.art.relation_coltypes.get(key)
        if typemap:
            return typemap.get(col.lower())
        # lowercase key fallback
        typemap2 = self.art.relation_coltypes.get(key.lower())
        if typemap2:
            return typemap2.get(col.lower())
        return None

    def relation_stats(self, relation_sql: str) -> Dict[str, Any]:
        key = sanitize_relation(relation_sql)
        return self.art.relation_stats.get(key, {})
    
    def _group_keys_df(self, uid: str, scope: Scope) -> List[DFExpr]:
        # Convert GROUP BY expressions to DFExprs (path-local seen set per key)
        return [self._expr_to_df(uid, scope, g, seen=set()) for g in scope.group_by_exprs]
    
    def _get_or_build_scope(self, uid: str) -> Optional[Scope]:
        node = self.art.nodes_by_uid[uid]
        sql = node.compiled_sql or node.raw_sql
        if not sql:
            return None

        scope_key = uid
        cached_scope = self._uid_scope_cache.get(scope_key)
        if cached_scope is not None:
            # debug
            # print(f"Using cached scope for model uid: {uid}")
            return cached_scope
        
        # debug
        # print(f"Building scope for model uid: {uid}, sql sig: {sig}")
        tree = sqlglot.parse_one(sql, read=self.dialect)
        scope = self.scope_builder.build_scope(tree, name=f"{uid}")
        self._uid_scope_cache[scope_key] = scope
        
        visited_scopes: Set[int] = set()
        for sc in self._walk_scopes_within_model(scope):
            if sc.scope_id in visited_scopes:
                continue
            visited_scopes.add(sc.scope_id)
            # print(f"Caching full scope for model uid: {uid}, scope id: {sc.scope_id}, scope name: {sc.scope_name}")
            # if uid not in sc.scope_name:
            #     print(f"Warning: scope name {sc.scope_name} does not contain uid {uid}")
            
            self._full_scope_cache[(uid, sc.scope_id, sc.scope_name)] = sc
        
        return scope

    def _collect_transitive_upstream_uids(self, unique_ids: List[str]) -> Set[str]:
        all_uids = set(unique_ids)
        queue = list(unique_ids)
        while queue:
            uid = queue.pop()
            node = self.art.nodes_by_uid.get(uid)
            if node is None:
                continue
            for parent_uid in node.depends_on_nodes:
                if parent_uid not in all_uids and parent_uid in self.art.nodes_by_uid:
                    all_uids.add(parent_uid)
                    queue.append(parent_uid)
        return all_uids

    @dataclass
    class _ScopeSig:
        uid: str
        scope_id: int
        scope_name: str
        tokens: Dict[str, float]    # weighted feature multiset
        anchors: Set[str]           # high-signal tokens for candidate generation
        
    
    
    def dfexpr_is_expensive(self, dfc: "DFExpr") -> bool:
        """
        Expensive iff dfc contains:
        - aggregation: DFTransform(op="agg")
        - window function: DFTransform(op in WINDOW_OPS)
        """
        WINDOW_OPS = {
            # adjust to your actual DFTransform.op vocabulary
            "window", "over",
            "row_number", "rank", "dense_rank", "ntile",
            "lag", "lead",
            "first_value", "last_value", "nth_value",
        }

        stack = [dfc]
        while stack:
            n = stack.pop()

            if isinstance(n, DFTransform):
                op = (n.op or "").lower()
                if op == "agg" or op in WINDOW_OPS:
                    return True

                # push children
                if op == "case":
                    for cond, val in (n.attrs.get("branches") or []):
                        stack.append(val)
                        stack.append(cond)
                    else_expr = n.attrs.get("else")
                    if else_expr is not None:
                        stack.append(else_expr)
                else:
                    for a in reversed(n.args or []):
                        stack.append(a)

        return False

    @staticmethod
    def _referenced_source_aliases(pred: exp.Expression, known_aliases: Set[str]) -> Set[str]:
        """Return the set of source aliases referenced by columns in *pred*."""
        aliases = set()
        for col in pred.find_all(exp.Column):
            tbl = col.table
            if tbl and tbl in known_aliases:
                aliases.add(tbl)
        return aliases

    def _get_scope_local_feature_fingerprint(self, uid: str, sc: Scope) -> Dict[str, Any]:
        """
        Cache expensive per-scope computations that are reused many times:
        - join fingerprints in this scope
        - group-by key fingerprints in this scope
        """
        key = (uid, sc.scope_id)
        if key in self._scope_local_fingerprint_cache:
            return self._scope_local_fingerprint_cache[key]
        # ---- Normalize predicates between WHERE and JOIN ON ----
        # Classify each conjunct by how many source aliases it references:
        #   2+ aliases  →  join condition
        #   0–1 aliases →  filter condition
        # Only WHERE predicates are candidates for reclassification;
        # HAVING / QUALIFY stay in filters unconditionally.

        known_aliases = set(sc.sources.keys())

        # Phase A: collect and classify WHERE conjuncts
        where_as_join: List[exp.Expression] = []
        where_as_filter: List[exp.Expression] = []
        kept_filters: List[Tuple[str, exp.Expression]] = []  # HAVING / QUALIFY

        for tag, pred in (sc.filters or []):
            if tag != "WHERE":
                kept_filters.append((tag, pred))
                continue
            conj = list(pred.flatten()) if isinstance(pred, exp.And) else [pred]
            for p in conj:
                aliases = self._referenced_source_aliases(p, known_aliases)
                if len(aliases) >= 2:
                    where_as_join.append(p)
                else:
                    where_as_filter.append(p)

        # Phase B: build local join expressions (no mutation of sc)
        local_joins: List[Tuple[str, Optional[exp.Expression]]] = []
        extra_filters_from_joins: List[exp.Expression] = []

        for i, j in enumerate(sc.joins or []):
            if j.on_expr is None:
                local_joins.append((j.jtype, None))
                continue
            conj = list(j.on_expr.flatten()) if isinstance(j.on_expr, exp.And) else [j.on_expr]
            join_keep = []
            for p in conj:
                aliases = self._referenced_source_aliases(p, known_aliases)
                if len(aliases) >= 2:
                    join_keep.append(p)
                else:
                    extra_filters_from_joins.append(p)
            # Append relocated WHERE conjuncts to the first join
            if i == 0:
                join_keep.extend(where_as_join)
            on_expr = exp.and_(*join_keep) if len(join_keep) > 1 else (join_keep[0] if join_keep else None)
            local_joins.append((j.jtype, on_expr))

        # If no joins exist but multi-alias WHERE conjuncts found, create a synthetic entry
        if not sc.joins and where_as_join:
            on_expr = exp.and_(*where_as_join) if len(where_as_join) > 1 else where_as_join[0]
            local_joins.append(("INNER", on_expr))

        # Phase C: build local filter list
        all_filter_conjuncts = where_as_filter + extra_filters_from_joins
        local_filters: List[Tuple[str, exp.Expression]] = list(kept_filters)
        if all_filter_conjuncts:
            combined = exp.and_(*all_filter_conjuncts) if len(all_filter_conjuncts) > 1 else all_filter_conjuncts[0]
            local_filters.append(("WHERE", combined))

        # ---- Fingerprint joins ----
        join_fps: List[str] = []
        for jtype, on_expr in local_joins:
            if on_expr is None:
                continue
            df_bool = self._expr_to_df(uid, sc, on_expr, seen=set())
            join_df = JoinDF(
                scope_id=sc.scope_id,
                scope_name=sc.scope_name,
                jtype=jtype,
                expr=df_bool,
            )
            join_fps.append(fingerprint_join(join_df))

        # ---- Group-by keys (per-key) ----
        gk_sigs: List[str] = []
        for dfg in self._group_keys_df(uid, sc):
            gk_sigs.append(fingerprint_dfexpr(dfg))

        gk_set_tok = None
        if gk_sigs:
            gk_set_tok = f"gk_set:{','.join(sorted(gk_sigs))}"

        # ---- Predicate features ----
        pred_fps: List[str] = []
        for tag, pred in local_filters:
            conj = list(pred.flatten()) if isinstance(pred, exp.And) else [pred]
            for p in conj:
                df_pred = self._expr_to_df(uid, sc, p, seen=set())
                pred_fps.append(fingerprint_dfexpr(df_pred))


        out = {
            "join_fps": join_fps,
            "pred_fps": pred_fps,
            "gk_sigs": gk_sigs,
            "gk_set_tok": gk_set_tok,
        }
        self._scope_local_fingerprint_cache[key] = out
        return out
    
    
    def _add_token(self, d: Dict[str, float], tok: str, w: float = 1.0):
        d[tok] = d.get(tok, 0.0) + float(w)

    # def _tokens_for_scope(self, uid, sc: Scope, w_join_edge=3.0, w_group_key=2.0) -> "_ScopeSig":
    def _tokens_for_scope(
        self,
        uid,
        sc: Scope,
        *,
        w_join_edge=3.0,
        w_group_key=3.0,
        w_pred=1.0,
        w_proj=4.0,
        include_context_from_model: bool = True,  # at least include local + context joins
        include_group_keys_from_context: bool = True,
        upstream_decay: float = 0.6          # deeper/nested scopes => lower weight
        
    ) -> "_ScopeSig":
        tokens: Dict[str, float] = {}
        anchors: Set[str] = set()
        
        def add_tok(tok: str, w: float, anchor: bool = False):
            self._add_token(tokens, tok, w)
            if anchor:
                anchors.add(tok)

        root = sc if include_context_from_model else None
        scope_iter = [(sc, 0)]

        if root is not None:
            scope_iter = list(self._walk_scopes_across_model_with_depth(root))
        
        # --- Joins (local + context with decay) ---
        for s2, d2 in scope_iter:
            # Weight multiplier
            if s2.scope_id == sc.scope_id:
                mult = 1.0
            else:
                # If s2 is deeper than sc, it's "upstream relative to sc".
                depth_diff = max(0, d2)
                mult = (upstream_decay ** depth_diff) if depth_diff > 0 else 1.0
                # If s2 is not deeper, we keep mult=1.0 (downstream/context is important)
                
            feats = self._get_scope_local_feature_fingerprint(uid, s2)
            for join_fp in feats["join_fps"]:
                tok = f"join:{join_fp}"
                add_tok(tok, w_join_edge * mult, anchor=True)


        # --- GROUP BY keys: per-key tokens (partial match works naturally) ---
        def add_group_keys_from_scope(s2: Scope, mult: float):
            feats = self._get_scope_local_feature_fingerprint(uid, s2)

            for sig in feats["gk_sigs"]:
                add_tok(f"gk:{sig}", w_group_key * mult, anchor=True)

            if feats["gk_set_tok"]:
                add_tok(feats["gk_set_tok"], 0.1 * w_group_key * mult, anchor=False)

        # local group keys
        add_group_keys_from_scope(sc, mult=1.0)

        # context group keys (optional)
        if include_context_from_model and include_group_keys_from_context and root is not None:
            for s2, d2 in scope_iter:
                if s2.scope_id == sc.scope_id:
                    continue
                mult = (upstream_decay ** max(0, d2))
                add_group_keys_from_scope(s2, mult=mult)

        # --- Projections: keep as tokens (you already downselect expensive ones) ---
        for col in (sc.proj_order or []):
            dfc = self._trace_projection_expr(uid=uid, scope=sc, proj_name=col, seen=set())
            if not self.dfexpr_is_expensive(dfc):
                continue
            tok = f"proj_col:{fingerprint_dfexpr(dfc)}"
            add_tok(tok, w_proj, anchor=False)
            
        # --- Predicates (usually keep local only; context preds can explode noise) ---
        # local predicates
        for tag, pred in (sc.filters or []):
            df_bool = self._expr_to_df(uid, sc, pred, seen=set())
            pred_df = PredicateDF(scope_id=sc.scope_id, scope_name=sc.scope_name, tag=tag, expr=df_bool)
            tok = f"pred:{fingerprint_predicate(pred_df)}"
            add_tok(tok, w_pred, anchor=False)

        return Extractor._ScopeSig(
            uid=uid,
            scope_id=sc.scope_id,
            scope_name=sc.scope_name,
            tokens=tokens,
            anchors=anchors,
        )



    # ---------------- Public API ----------------
    
    def group_models_by_scope_similarity(self,
                                         unique_ids: List[str],
                                         *,
                                         max_anchor_dist_allowed: float = 0,
                                         top_n_scope_pairs: int = 2000,
                                         min_edge_weight: float = 0.6,
                                         min_degree_k: int = 1,
                                         min_models_in_group: int = 2,
                                         ) -> List[List[str]]:
        
        # Parameters
        min_anchor_sim = 1.0 - max_anchor_dist_allowed
        
        sigs = []
        for uid in unique_ids:
            _ = self._get_or_build_scope(uid)

        
        items = list(self._full_scope_cache.items())
        for (uid, _, _), sc in items:
            sig = self._tokens_for_scope(uid, sc)
            sigs.append(sig)

            
        if not sigs:
            print("No scope signatures found.")
            return []
        
        # Build anchor-weight dicts
        anchor_w = []
        for s in sigs:
            aw = {}
            for a in s.anchors:
                w = s.tokens.get(a, 0.0)
                if w > 0:
                    aw[a] = w
            anchor_w.append(aw)
            
        anchor_sum = [sum(aw.values()) for aw in anchor_w]

        # -----------------------------
        # Step 1: generate candidate scope pairs by shared anchors
        # -----------------------------
        inv = defaultdict(list)  # anchor -> [scope_idx]
        for i, s in enumerate(sigs):
            for a in s.anchors:
                inv[a].append(i)

        # -----------------------------
        # Step 1b: compute weighted anchor similarity for candidates
        # -----------------------------
        heap = []  # keep top scope pairs by FULL similarity: (sim, anchor_sim, i, j)

        for i, _ in enumerate(sigs):
            # accumulate weighted overlap on anchors: sum of min(wi, wj) for shared anchors
            overlap = defaultdict(float)
            for a, wi in anchor_w[i].items():
                for j in inv[a]:
                    if j <= i:
                        continue
                    wj = anchor_w[j].get(a, 0.0)
                    if wj > 0:
                        overlap[j] += min(wi, wj)

            # compute anchor_sim for each candidate j
            sum_i = anchor_sum[i]
            for j, inter in overlap.items():
                sum_j = anchor_sum[j]
                union = sum_i + sum_j - inter
                if union <= 0:
                    continue
                a_sim = inter / union  # weighted jaccard on anchors
                if a_sim < min_anchor_sim:
                    continue

                # Step 2: full similarity on tokens
                sim = weighted_jaccard(sigs[i].tokens, sigs[j].tokens)

                item = (sim, a_sim, i, j)
                if len(heap) < top_n_scope_pairs:
                    heapq.heappush(heap, item)
                else:
                    if item > heap[0]:
                        heapq.heapreplace(heap, item)

        top = sorted(heap, reverse=True)

        # -----------------------------
        # Step 3: aggregate scope pairs -> model edges (max)
        # -----------------------------
        edge_max = defaultdict(float)
        for sim, a_sim, i, j in top:
            ui, uj = sigs[i].uid, sigs[j].uid
            if ui == uj:
                continue
            a, b = (ui, uj) if ui < uj else (uj, ui)
            if sim > edge_max[(a, b)]:
                edge_max[(a, b)] = sim

        # thresholded model graph
        adj = defaultdict(set)
        for (a, b), w in edge_max.items():
            if w >= min_edge_weight:
                adj[a].add(b)
                adj[b].add(a)
        
        
        # for debugging / info
        num_scope_pairs = len(top)
        num_model_pairs = len(edge_max)
        num_model_pairs_pass_weight_threshold = sum(1 for w in edge_max.values() if w >= min_edge_weight)
        
        # number of unique models involved
        unique_models_involved = set()
        for (a, b) in edge_max.keys():
            unique_models_involved.add(a)
            unique_models_involved.add(b)

        # print(f"scope pairs: {num_scope_pairs}")
        # print(f"model pairs: {num_model_pairs}")
        # print(f"model pairs passing edge weight threshold ({min_edge_weight}): {num_model_pairs_pass_weight_threshold}")
        # print(f"unique models involved: {len(unique_models_involved)}")
        # ==============================================
                

        # connected components
        visited = set()
        comps = []
        for node in list(adj.keys()):
            if node in visited:
                continue
            stack = [node]
            visited.add(node)
            comp = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for nb in adj[x]:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            comps.append(comp)

        groups = []
        for comp in comps:
            # when min_degree_k > 1, extract k-core
            # otherwise, use the original connected component
            core = k_core(comp, min_degree_k, adj) if min_degree_k > 1 else sorted(comp)
            if len(core) < min_models_in_group:
                continue
            groups.append(core)
            
            # n = len(core)
            # e = induced_edges(core, adj)
            # dens = (2 * e) / (n * (n - 1)) if n > 1 else 0.0
            # if dens >= min_density:
            #     groups.append(core)
            
            # print(f"Component size: {len(comp)}, core size: {len(core)}, edges: {e}, density: {dens:.4f}")

        groups.sort(key=len, reverse=True)
        
        # number of uids covered
        uids_covered = set()
        for g in groups:
            for uid in g:
                uids_covered.add(uid)
        print(f"Number of unique models covered in groups: {len(uids_covered)} out of {len(unique_ids)}")
        
        
        # For fast membership checks
        groups_fs = [set(g) for g in groups]

        # Track the latest (max) position in `top` of any edge whose endpoints are both in the group
        last_pos_in_group = [-1] * len(groups_fs)

        for pos, (sim, a_sim, i, j) in enumerate(top):
            ui, uj = sigs[i].uid, sigs[j].uid
            if ui == uj:
                continue

            # update any group that contains both endpoints
            for gi, gset in enumerate(groups_fs):
                if ui in gset and uj in gset:
                    last_pos_in_group[gi] = pos  # since pos increases, overwrite is fine

        for gi, g in enumerate(groups):
            print(f"group#{gi} size={len(g)} last_internal_edge_pos={last_pos_in_group[gi]}")
        
        return groups


    '''
        scope_backbone_accum_dag = {}
        backbone_accum_dag_scope_list = defaultdict(list)  # backbone signature -> list of (uid, scope_id, scope_name)
        
        scope_iter_dag = [(sc, 0)]
        scope_iter_dag = list(self._walk_scopes_across_model_with_depth(root))
        
        bb_dag = set()
        for s2, _ in scope_iter_dag:
            s2_uid = s2.scope_name.split(':', 1)[0]
            ident = (s2_uid, s2.scope_id, s2.scope_name)
            local_bb = local_bb_dict.get(ident, set())
            bb_dag.update(local_bb)
        bb_dag = frozenset(bb_dag)
        scope_backbone_accum_dag[this_ident] = bb_dag
        backbone_accum_dag_scope_list[bb_dag].append(this_ident)
    
    '''





















    def generate_skeleton_features(self, unique_ids: List[str], dbt_run_id: str):
        scope_backbone_accum_model = {}
        backbone_accum_model_scope_list = defaultdict(list)  # backbone signature -> list of (uid, scope_id, scope_name)

        all_uids = self._collect_transitive_upstream_uids(unique_ids)

        for uid in all_uids:
            _ = self._get_or_build_scope(uid)
            
        
        items = list(self._full_scope_cache.items())
        local_bb_dict = {}
        
        print(f"Total scopes across all models: {len(items)}")
        
        for (uid, _, _), sc in items:
            local_bb_set = set()
            feats = self._get_scope_local_feature_fingerprint(uid, sc)
            # join
            for join_fp in feats["join_fps"]:
                tok = f"join:{join_fp}"
                local_bb_set.add(tok)
            # agg
            if feats["gk_set_tok"]:
                local_bb_set.add(feats["gk_set_tok"])
                
            # predicate
            for pred_fp in feats["pred_fps"]:
                tok = f"pred:{pred_fp}"
                local_bb_set.add(tok)
                
                
            local_bb_dict[(uid, sc.scope_id, sc.scope_name)] = local_bb_set
        
        
        local_fanout_dict = Counter()
        for (uid, _, _), sc in items:
            this_ident = (uid, sc.scope_id, sc.scope_name)
            root = sc
            scope_iter_model = [sc]
            if root is not None:
                scope_iter_model = list(self._walk_scopes_within_model(root))
                
                if uid == sc.scope_name:
                    for s2 in scope_iter_model:
                        s2_ident = (uid, s2.scope_id, s2.scope_name)
                        local_fanout_dict[s2_ident] += 1

            
            bb_model = Counter()
            for s2 in scope_iter_model:
                s2_uid = s2.scope_name.split(':', 1)[0]
                ident = (s2_uid, s2.scope_id, s2.scope_name)

                local_bb = local_bb_dict.get(ident, set())  # local_bb is a set
                bb_model.update(local_bb)                   # each token count += 1

            # freeze multiset into a hashable key for dicts
            bb_model_key = tuple(sorted(bb_model.items()))  # e.g. (("join:x", 3), ("gk:...", 1))

            # store counts (Counter) or store the frozen key; pick what you prefer
            scope_backbone_accum_model[this_ident] = bb_model          # keeps multiplicities
            backbone_accum_model_scope_list[bb_model_key].append(this_ident)
        

        
        too_simple_threshold = 1
        backbone_accum_model_scope_list = {bb: scopes for bb, scopes in backbone_accum_model_scope_list.items() if len(bb) >= too_simple_threshold}
        
        # Remove scopes whose scope_name contains 'union_child'
        backbone_accum_model_scope_list = {
            bb: [t for t in scopes if 'union_child' not in t[2]]  # t[2] == scope_name
            for bb, scopes in backbone_accum_model_scope_list.items()
        }
        # remove backbones that are shared by no scopes after filtering
        backbone_accum_model_scope_list = {bb: scopes for bb, scopes in backbone_accum_model_scope_list.items() if scopes}
        
        
        # find common backbones shared by multiple models
        
        bb_model_list = backbone_accum_model_scope_list.copy()
        # only keep scopes that uid is the same as scope name
        bb_model_list = {
            bb: [t[0] for t in scopes if t[0] == t[2]]
            for bb, scopes in bb_model_list.items()
        }
        
        
        # output a json file
        # key is uid, value is the dict {token: frequency} for that model's backbone
        backbone_json = {}
        for (uid, scope_id, scope_name), bb_model in scope_backbone_accum_model.items():
            if uid != scope_name:
                continue
            backbone_json[uid] = bb_model
            
        # output_file_name = 'skeleton_files/original_skeleton.json'
        output_file_name = f'skeleton_files/{dbt_run_id}_skeleton.json'
        with open(output_file_name, 'w') as f:
            json.dump(backbone_json, f, indent=2) 
        














    # yyyyyy
    def analysis_for_reuse(self, unique_ids: List[str]):
        scope_backbone_accum_model = {}
        backbone_accum_model_scope_list = defaultdict(list)  # backbone signature -> list of (uid, scope_id, scope_name)
        
        all_uids = self._collect_transitive_upstream_uids(unique_ids)
        
        for uid in all_uids:
            _ = self._get_or_build_scope(uid)
        
        items = list(self._full_scope_cache.items())
        local_bb_dict = {}
        
        for (uid, _, _), sc in items:
            local_bb_set = set()
            feats = self._get_scope_local_feature_fingerprint(uid, sc)
            # join
            for join_fp in feats["join_fps"]:
                tok = f"join:{join_fp}"
                local_bb_set.add(tok)
            # agg
            if feats["gk_set_tok"]:
                local_bb_set.add(feats["gk_set_tok"])
                
            # predicate
            # for pred_fp in feats["pred_fps"]:
            #     tok = f"pred:{pred_fp}"
            #     local_bb_set.add(tok)
                
                
            local_bb_dict[(uid, sc.scope_id, sc.scope_name)] = local_bb_set
        
        
        local_fanout_dict = Counter()
        for (uid, _, _), sc in items:
            this_ident = (uid, sc.scope_id, sc.scope_name)
            root = sc
            scope_iter_model = [sc]
            if root is not None:
                scope_iter_model = list(self._walk_scopes_within_model(root))
                
                if uid == sc.scope_name:
                    for s2 in scope_iter_model:
                        s2_ident = (uid, s2.scope_id, s2.scope_name)
                        local_fanout_dict[s2_ident] += 1

            
            bb_model = Counter()
            for s2 in scope_iter_model:
                s2_uid = s2.scope_name.split(':', 1)[0]
                ident = (s2_uid, s2.scope_id, s2.scope_name)

                local_bb = local_bb_dict.get(ident, set())  # local_bb is a set
                bb_model.update(local_bb)                   # each token count += 1

            # freeze multiset into a hashable key for dicts
            bb_model_key = tuple(sorted(bb_model.items()))  # e.g. (("join:x", 3), ("gk:...", 1))

            # store counts (Counter) or store the frozen key; pick what you prefer
            scope_backbone_accum_model[this_ident] = bb_model          # keeps multiplicities
            backbone_accum_model_scope_list[bb_model_key].append(this_ident)
        

        
        too_simple_threshold = 1
        backbone_accum_model_scope_list = {bb: scopes for bb, scopes in backbone_accum_model_scope_list.items() if len(bb) >= too_simple_threshold}
        
        # Remove scopes whose scope_name contains 'union_child'
        backbone_accum_model_scope_list = {
            bb: [t for t in scopes if 'union_child' not in t[2]]  # t[2] == scope_name
            for bb, scopes in backbone_accum_model_scope_list.items()
        }
        # remove backbones that are shared by no scopes after filtering
        backbone_accum_model_scope_list = {bb: scopes for bb, scopes in backbone_accum_model_scope_list.items() if scopes}
        
        
        
        
        # find common backbones shared by multiple models
        
        bb_model_list = backbone_accum_model_scope_list.copy()
        # only keep scopes that uid is the same as scope name
        bb_model_list = {
            bb: [t[0] for t in scopes if t[0] == t[2]]
            for bb, scopes in bb_model_list.items()
        }
        
        '''
        # output a json file
        # key is uid, value is the dict {token: frequency} for that model's backbone
        backbone_json = {}
        for (uid, scope_id, scope_name), bb_model in scope_backbone_accum_model.items():
            if uid != scope_name:
                continue
            backbone_json[uid] = bb_model
            
        # output_file_name = 'skeleton_files/original_skeleton.json'
        output_file_name = 'skeleton_files/iter_5_skeleton.json'
        with open(output_file_name, 'w') as f:
            json.dump(backbone_json, f, indent=2) 
        '''
        
        
        
        # remove backbones that are shared by no scopes after filtering
        bb_model_list = {bb: scopes for bb, scopes in bb_model_list.items() if scopes}
        
        
        def _intersect_bb(bb1, bb2):
            c1, c2 = Counter(dict(bb1)), Counter(dict(bb2))
            common = {k: min(c1[k], c2[k]) for k in (c1.keys() & c2.keys())}
            return tuple(sorted((k, v) for k, v in common.items() if v > 0))
        
        def _is_subset_bb(bb_small, bb_large):
            """
            Multiset subset check:
            bb_small ⊆ bb_large iff for every token t, cnt_small[t] <= cnt_large[t]
            """
            c_small = Counter(dict(bb_small))
            c_large = Counter(dict(bb_large))
            for k, v in c_small.items():
                if c_large.get(k, 0) < v:
                    return False
            return True
        
        def mass_from_bb_key(bb_key):
            return sum(cnt for _, cnt in bb_key)
        
        common_bb_dict = defaultdict(set) # common_bb: set of backbones that share it
        # bb = backbone signature (tuple of (token, count)), ml = list of uids with this backbone
        for (bb1, ml1), (bb2, ml2) in combinations(bb_model_list.items(), 2):
            common_bb = _intersect_bb(bb1, bb2)
            if common_bb:
                common_bb_dict[common_bb].add(bb1)
                common_bb_dict[common_bb].add(bb2)
                
        # propagate: if common_bb_1 ⊆ common_bb_2, then add bbs_2 into bbs_1
        # (i.e., the more general/smaller common pattern inherits supporting backbones from more specific ones)
        common_keys = list(common_bb_dict.keys())
        size_cache = {bb: sum(cnt for _, cnt in bb) for bb in common_keys}


        for i, common_bb_1 in enumerate(common_keys):
            for j, common_bb_2 in enumerate(common_keys):
                if i == j:
                    continue
                if size_cache[common_bb_1] > size_cache[common_bb_2]:
                    continue
                if _is_subset_bb(common_bb_1, common_bb_2):
                    common_bb_dict[common_bb_1].update(common_bb_dict[common_bb_2])
        
        
        common_bb_dict_plus = common_bb_dict.copy()
        # add all bbs that in bb_model_list but not in common_bb_dict, as they are also "common" in the sense that they are shared by multiple scopes (even if they don't have a smaller common subset)
        for bb in bb_model_list.keys():
            if bb not in common_bb_dict_plus:
                common_bb_dict_plus[bb] = set([bb])
                
        # compute score for each common_bb
        all_groups = []
        for common_bb, bbs in common_bb_dict_plus.items():
            related_models = set(uid for bb in bbs for uid in bb_model_list.get(bb, []))
            num_models = len(related_models)
            mass = mass_from_bb_key(common_bb)
            score = mass * num_models
                
            all_groups.append((common_bb, bbs, related_models, mass, score))
                
        all_groups.sort(key=lambda x: x[4], reverse=True)  # sort by score
        
        
        top_k_groups = 30
        
        output = {}
        for rank, (common_bb, bbs, related_models, mass, score) in enumerate(all_groups[:top_k_groups]):
            group_key = f"group_{rank}"
            output[group_key] = {
                "score": score,
                "mass": mass,
                "num_models": len(related_models),
                "models": sorted(related_models),
            }

        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "examples", "outputs", "0324_dataflow_analysis_top_groups.json"
        )
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        return output
        
        
        
        
        
        '''
        
        # compute pairwise Jaccard similarity between backbones
        l = []
        for bb1, scopes1 in bb_model_list.items():
            for bb2, scopes2 in bb_model_list.items():
                if bb1 >= bb2:  # avoid duplicate pairs and self-comparison
                    continue
                sim, common_counter, inter_num = backbone_similarity(bb1, bb2)
                if sim > 0.0:  # example threshold for "similar" backbones
                    
                    # if their models are the same, then skip
                    model_names1 = set(t[0] for t in scopes1)
                    model_names2 = set(t[0] for t in scopes2)
                    if model_names1 == model_names2:
                        continue
                    
                    
                    l.append((sim, inter_num, common_counter, bb1, scopes1, bb2, scopes2))
                    
        # sort by similarity
        l.sort(key=lambda x: x[0], reverse=True)
        for sim, inter_num, common_counter, bb1, scopes1, bb2, scopes2 in l:
            print(f"Similar backbones with similarity {sim:.2f}, common: {inter_num}")
            for tok, cnt in common_counter.most_common(10):
                print(f"    {tok} (count: {cnt})")
                
            # based on common_counter, you can also print the unique tokens in each backbone to understand differences
            unique_to_bb1 = Counter(dict(bb1)) - common_counter
            unique_to_bb2 = Counter(dict(bb2)) - common_counter
            print(f"  Unique to Backbone 1: {len(unique_to_bb1)} tokens")
            for tok, cnt in unique_to_bb1.most_common(10):
                print(f"    {tok} (count: {cnt})")
            print(f"  Unique to Backbone 2: {len(unique_to_bb2)} tokens")
            for tok, cnt in unique_to_bb2.most_common(10):
                print(f"    {tok} (count: {cnt})")
                
            print(f"  Backbone 1: size {len(bb1)} shared by {len(scopes1)} scopes")
            for scope in scopes1:
                print(f"    {scope[2]}")
            # for bb, cnt in bb1:
            #     print(f"    {bb} (count: {cnt})")
            print(f"  Backbone 2: size {len(bb2)} shared by {len(scopes2)} scopes")
            for scope in scopes2:
                print(f"    {scope[2]}")
            # for bb, cnt in bb2:
            #     print(f"    {bb} (count: {cnt})")
            print("")
            
        
        '''
        
        
        
        
        '''
        # -------------------------------
        # (1) PRIORITIZE FAMILIES
        # family_key := frozenset(models in the group)
        # family_score := max_G mass(G) * |Models(G)|
        # mass(G) := total feature multiplicity (equal weights)
        # -------------------------------
        def mass_from_bb_key(bb_key):
            return sum(cnt for _, cnt in bb_key)

        family_best = {}
        # family_key -> (best_score, best_bb_key, best_scopes)

        for bb_key, scopes in backbone_accum_model_scope_list.items():
            models = frozenset(uid for (uid, _, _) in scopes)
            M = len(models)
            if M == 0:
                continue
            score = mass_from_bb_key(bb_key) * M

            prev = family_best.get(models)
            if (prev is None) or (score > prev[0]):
                family_best[models] = (score, bb_key, scopes)

        families_ranked = sorted(family_best.items(), key=lambda kv: kv[1][0], reverse=True)
        # families_ranked = families_ranked[:top_k_families]

        # -------------------------------
        # (2) CUT POINTS INSIDE MODELS via mat_score
        # mat_score(scope) = local_mass(scope) * (fanout(scope) - 1)
        # -------------------------------
        results = []
        top_k_cutpoints_per_model = 3
        for family_models, (fam_score, dominant_bb_key, dominant_scopes) in families_ranked:
            cutpoints_by_model: Dict[str, List[Dict[str, Any]]] = {}

            for (uid, sid, sname), fanout in local_fanout_dict.items():
                if uid not in family_models:
                    continue
                if "union_child" in sname:
                    continue

                local_mass = len(local_bb_dict.get((uid, sid, sname), set()))
                if local_mass <= 0:
                    continue
                if fanout <= 1:
                    continue

                mat_score = local_mass * (fanout - 1)
                cutpoints_by_model.setdefault(uid, []).append({
                    "mat_score": mat_score,
                    "fanout": fanout,
                    "local_mass": local_mass,
                    "scope": (uid, sid, sname),
                })

            # keep top-k per model
            for uid, lst in cutpoints_by_model.items():
                lst.sort(key=lambda d: d["mat_score"], reverse=True)
                lst_len = len(lst)
                if lst_len > top_k_cutpoints_per_model:
                    cutpoints_by_model[uid] = lst[:top_k_cutpoints_per_model]
                else:
                    cutpoints_by_model[uid] = lst

            results.append({
                "family_models": tuple(sorted(family_models)),
                "family_score": fam_score,
                "dominant_group": {
                    "mass": mass_from_bb_key(dominant_bb_key),
                    "num_models": len(family_models),
                    "bb_key": dominant_bb_key,
                    "scopes": dominant_scopes,
                },
                "cutpoints_by_model": cutpoints_by_model,
            })
            
        # print top families and cut points for debugging
        for i, fam in enumerate(results):
            print(f"Family #{i+1}: family_score={fam['family_score']:.2f}")
            for uid in fam["family_models"]:
                print(f"  {uid}")
            print(f"  Dominant group: mass={fam['dominant_group']['mass']}, num_models={fam['dominant_group']['num_models']}, bb_key={fam['dominant_group']['bb_key']}")
            for uid, cutpoints in fam["cutpoints_by_model"].items():
                print(f"    Model {uid} cut points:")
                for cp in cutpoints:
                    print(f"      Scope: {cp['scope']}, mat_score: {cp['mat_score']:.2f}, local_mass: {cp['local_mass']}, fanout: {cp['fanout']}")
            print("")
            
        '''

        





























    def _iter_column_refs(self, e: Optional[exp.Expression]):
        if e is None:
            return
        for c in e.find_all(exp.Column):
            # Skip stars like * or t.*
            try:
                if c.name == "*":
                    continue
            except Exception:
                pass
            yield c

    def _exprs_relevant_for_scope_demand(
        self,
        scope: Scope,
        demanded_outputs: Set[str],
        force_all_proj: bool = False,
    ) -> List[Tuple[str, exp.Expression]]:
        """
        Return expressions whose input columns are needed to preserve semantics
        for this scope given demanded outputs.
        """
        exprs: List[Tuple[str, exp.Expression]] = []

        # Projections: only demanded outputs (unless force_all_proj)
        if force_all_proj:
            proj_names = scope.proj_order
        else:
            proj_names = [p for p in scope.proj_order if p in demanded_outputs]

        for p in proj_names:
            pe = scope.projections.get(p)
            if pe is not None:
                exprs.append((f"PROJ:{p}", pe))

        # Filters/grouping/joins affect semantics of outputs -> always include
        for tag, f in scope.filters:
            if f is not None:
                exprs.append((tag, f))

        for j in scope.joins:
            if j.on_expr is not None:
                exprs.append((f"JOIN_ON:{j.jtype}", j.on_expr))

        for g in scope.group_by_exprs:
            exprs.append(("GROUP_BY", g))

        return exprs

    def _source_exposed_cols(self, uid: str, scope: Scope, alias: str) -> Optional[Set[str]]:
        """
        Columns available from source alias inside this scope.
        Returns None if unknown (e.g., stars / missing catalog).
        """
        if alias not in scope.sources:
            return None

        skind, sobj = scope.sources[alias]

        if skind == "table":
            rel = sobj
            key = sanitize_relation(rel)
            # Try exact catalog match
            cols = self.art.relation_columns.get(key)
            if cols:
                return set(cols)
            cols = self.art.relation_columns_lc.get(key.lower())
            if cols:
                return set(cols)
            # Fallback: database-agnostic catalog match (cross-env catalog/manifest)
            from .dbt_info import _strip_database
            no_db = _strip_database(key)
            if no_db:
                cols = self.art._relation_columns_no_db.get(no_db.lower())
                if cols:
                    return set(cols)
            # Fallback: use cached scope proj_order if we've already parsed the upstream model
            upstream_uid = self.art.relation_to_uid(rel)
            if upstream_uid:
                upstream_scope = self._uid_scope_cache.get(upstream_uid)
                if upstream_scope and not upstream_scope.has_star:
                    return set(upstream_scope.proj_order)
            return None  # unknown catalog coverage

        if skind in ("subquery", "cte"):
            child: Scope = sobj
            # If child has star, exact exposed cols may be unknown
            if child.has_star:
                return None
            return set(child.proj_order)

        return None

    def _resolve_unqualified_col_candidates(self, uid: str, scope: Scope, col_name: str) -> List[str]:
        """
        Resolve an unqualified column to candidate source aliases.
        Conservative: if ambiguous, return all matches.
        """
        col_l = col_name.lower()
        cands: List[str] = []

        for alias, (skind, sobj) in scope.sources.items():
            exposed = self._source_exposed_cols(uid, scope, alias)
            if exposed is None:
                # unknown exposure -> cannot rule out; to stay safe, treat as candidate
                cands.append(alias)
                continue

            exposed_l = {c.lower() for c in exposed}
            if col_l in exposed_l:
                cands.append(alias)
        return cands
    
    def _collect_local_source_usage_for_demand(
        self,
        uid: str,
        scope: Scope,
        demanded_outputs: Set[str],
        force_all_proj: bool = False,
         # {child_scope_name: {parent_scope_name: (set of columns used by this scope)}}
    ):

        local_used = defaultdict(set)
        exprs = self._exprs_relevant_for_scope_demand(scope, demanded_outputs, force_all_proj=force_all_proj)
        
        # build a dict for mapping alias to scope name
        alias_to_scope_name = {}
        for alias, (skind, sobj) in scope.sources.items():
            if skind in ("subquery", "cte") and isinstance(sobj, Scope):
                alias_to_scope_name[alias] = sobj.scope_name
            elif skind == "table" and isinstance(sobj, str):
                uid = self.art.uid_by_relation.get(sobj)
                if uid:
                    alias_to_scope_name[alias] = uid

        for tag, e in exprs:
            for c in self._iter_column_refs(e):
                col_name = c.name
                tbl = c.table  # sqlglot convenience property; may be None / ""

                if tbl:
                    alias = tbl
                    if alias in scope.sources:
                        upstream_scope_name = alias_to_scope_name.get(alias)
                        local_used[upstream_scope_name].add(col_name)
                    else:
                        print(f"Warning: in scope {scope.scope_name}, column {col_name} has table {alias} which is not in sources; treating as unqualified ({tag})")
                else:
                    cands = self._resolve_unqualified_col_candidates(uid, scope, col_name)
                    if not cands:
                        print(f"Warning: in scope {scope.scope_name}, unqualified column {col_name} has no candidate sources; possible missing catalog info or unsupported pattern ({tag})")
                    elif len(cands) == 1:
                        upstream_scope_name = alias_to_scope_name.get(cands[0])
                        local_used[upstream_scope_name].add(col_name)
                    else:
                        print(f"Warning: in scope {scope.scope_name}, unqualified column {col_name} has multiple candidate sources {cands}; assigning to all (tag: {tag})")
        return local_used

    def _collect_local_source_usage_by_tag(
        self,
        uid: str,
        scope: Scope,
        demanded_outputs: Set[str],
        force_all_proj: bool = False,
    ) -> Dict[str, Dict[str, Set[str]]]:
        """
        Like _collect_local_source_usage_for_demand, but buckets column references
        by tag category: "PROJ", "FILTER", "JOIN_ON", "GROUP_BY".
        Returns {tag_category: {upstream_scope_name: set(columns)}}.
        """
        result: Dict[str, Dict[str, Set[str]]] = {
            "PROJ": defaultdict(set),
            "FILTER": defaultdict(set),
            "JOIN_ON": defaultdict(set),
            "GROUP_BY": defaultdict(set),
        }
        exprs = self._exprs_relevant_for_scope_demand(scope, demanded_outputs, force_all_proj=force_all_proj)

        alias_to_scope_name = {}
        for alias, (skind, sobj) in scope.sources.items():
            if skind in ("subquery", "cte") and isinstance(sobj, Scope):
                alias_to_scope_name[alias] = sobj.scope_name
            elif skind == "table" and isinstance(sobj, str):
                resolved = self.art.uid_by_relation.get(sobj)
                if resolved:
                    alias_to_scope_name[alias] = resolved

        for tag, e in exprs:
            if tag.startswith("PROJ:"):
                category = "PROJ"
            elif tag in ("WHERE", "HAVING", "QUALIFY"):
                category = "FILTER"
            elif tag.startswith("JOIN_ON:"):
                category = "JOIN_ON"
            elif tag == "GROUP_BY":
                category = "GROUP_BY"
            else:
                category = "FILTER"  # conservative fallback

            for c in self._iter_column_refs(e):
                col_name = c.name
                tbl = c.table

                if tbl:
                    if tbl in scope.sources:
                        upstream_scope_name = alias_to_scope_name.get(tbl)
                        if upstream_scope_name:
                            result[category][upstream_scope_name].add(col_name)
                else:
                    cands = self._resolve_unqualified_col_candidates(uid, scope, col_name)
                    if len(cands) == 1:
                        upstream_scope_name = alias_to_scope_name.get(cands[0])
                        if upstream_scope_name:
                            result[category][upstream_scope_name].add(col_name)
                    else:
                        # ambiguous: assign to all candidates (conservative)
                        for cand in cands:
                            upstream_scope_name = alias_to_scope_name.get(cand)
                            if upstream_scope_name:
                                result[category][upstream_scope_name].add(col_name)

        return result

    @staticmethod
    def _extract_equijoin_pairs(on_expr: Optional[exp.Expression]) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Parse a JOIN ON expression and extract equi-join column pairs.
        Returns list of ((alias_a, col_a), (alias_b, col_b)) for each
        simple `alias_a.col_a = alias_b.col_b` equality.
        Only handles AND-connected equalities on plain column references.
        OR, functions, expressions are conservatively ignored.
        """
        if on_expr is None:
            return []

        pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
        # Collect all EQ nodes that are AND-connected (top-level or under AND)
        eq_nodes: List[exp.Expression] = []

        def _collect_eqs(node: exp.Expression):
            if isinstance(node, exp.EQ):
                eq_nodes.append(node)
            elif isinstance(node, exp.And):
                _collect_eqs(node.left)
                _collect_eqs(node.right)
            # OR, NOT, and other operators: stop recursing (conservative)

        _collect_eqs(on_expr)

        for eq in eq_nodes:
            left = eq.left
            right = eq.right
            # Both sides must be simple column references with table qualifiers
            if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                continue
            left_table = left.table
            right_table = right.table
            if not left_table or not right_table:
                continue  # unqualified columns — can't determine which source
            left_col = left.name
            right_col = right.name
            if left_col == "*" or right_col == "*":
                continue
            pairs.append(((left_table, left_col), (right_table, right_col)))

        return pairs

    @staticmethod
    def _classify_join_pruneability(join_type: str, has_star: bool) -> Tuple[str, str]:
        """
        Classify how safely a join can be removed when its source has
        zero non-join-ON usage (or fully substitutable usage).
        Returns (confidence, reason).
        """
        if has_star:
            return ("needs_verification",
                    "Scope uses SELECT *; cannot confirm unused columns. Manual review required.")

        jt = join_type.upper()
        if jt == "LEFT":
            return ("likely_safe",
                    "LEFT JOIN with unused right side. All left-side rows preserved. "
                    "Verify join is many:1 or 1:1 (no row fan-out on join keys).")
        elif jt == "RIGHT":
            return ("likely_safe",
                    "RIGHT JOIN with unused left side. All right-side rows preserved. "
                    "Verify join is 1:many or 1:1 (no row fan-out on join keys).")
        elif jt in ("INNER", ""):
            return ("needs_verification",
                    "INNER JOIN may filter rows even when no columns from this side are used. "
                    "Only safe if join condition guarantees all rows match (e.g., FK with no nulls).")
        elif jt == "CROSS":
            return ("needs_verification",
                    "CROSS JOIN changes row cardinality. Not safely removable without semantic analysis.")
        elif jt == "FULL":
            return ("needs_verification",
                    "FULL OUTER JOIN: removing one side changes NULL patterns in results.")
        else:
            return ("needs_verification",
                    f"Unknown join type '{jt}'. Manual review required.")

    def _collect_null_checked_source_cols(
        self,
        scope: Scope,
    ) -> Dict[str, Set[str]]:
        """
        Scan all expressions in a scope for IS NULL / IS NOT NULL checks on columns.
        Returns {source_alias: set(column_names)} for columns appearing in such checks.
        These columns carry NULL-semantics (e.g., testing whether a LEFT JOIN matched)
        and cannot be safely substituted via equi-join equivalences.
        """
        result: Dict[str, Set[str]] = defaultdict(set)

        # Collect all expressions: projections, filters, joins, group by
        all_exprs: List[exp.Expression] = []
        for _, pe in scope.projections.items():
            if pe is not None:
                all_exprs.append(pe)
        for _, fe in scope.filters:
            if fe is not None:
                all_exprs.append(fe)
        for ji in scope.joins:
            if ji.on_expr is not None:
                all_exprs.append(ji.on_expr)
        for ge in scope.group_by_exprs:
            all_exprs.append(ge)

        for expr in all_exprs:
            for is_node in expr.find_all(exp.Is):
                col_node = is_node.args.get("this")
                if isinstance(col_node, exp.Column):
                    tbl = col_node.table
                    col_name = col_node.name
                    if tbl and tbl in scope.sources and col_name != "*":
                        result[tbl].add(col_name.lower())

        return result

    def analysis_for_prune(self, unique_ids: List[str]) -> PruneAnalysisResult:
        """
        Detect potential prune opportunities:
        1. JOINs where the joined source contributes zero columns downstream
           (or only contributes columns substitutable via equi-join equivalences).
        2. Projected columns that are never consumed by any downstream scope.
        """
        import logging
        log = logging.getLogger(__name__)

        # --- Phase 1: Build scopes and dependency graph ---
        all_uids = self._collect_transitive_upstream_uids(unique_ids)
        for uid in all_uids:
            _ = self._get_or_build_scope(uid)

        scope_name_to_scope: Dict[str, Scope] = {}
        scope_name_to_uid: Dict[str, str] = {}
        for (uid, sc_id, sc_name), sc in self._full_scope_cache.items():
            scope_name_to_scope[sc_name] = sc
            scope_name_to_uid[sc_name] = uid

        # children_dict[X] = scopes that consume X (X is their upstream data source)
        # parent_dict[X] = scopes that provide data to X (X's upstream sources)
        children_dict: Dict[str, Set[str]] = defaultdict(set)
        parent_dict: Dict[str, Set[str]] = defaultdict(set)
        for (uid, sc_id, sc_name), sc in self._full_scope_cache.items():
            for alias, (skind, sobj) in sc.sources.items():
                if skind in ("subquery", "cte") and isinstance(sobj, Scope):
                    children_dict[sobj.scope_name].add(sc_name)
                    parent_dict[sc_name].add(sobj.scope_name)
                elif skind == "table" and isinstance(sobj, str):
                    resolved_uid = self.art.uid_by_relation.get(sobj)
                    if resolved_uid:
                        children_dict[resolved_uid].add(sc_name)
                        parent_dict[sc_name].add(resolved_uid)

        all_nodes = set(children_dict.keys()) | set(parent_dict.keys())
        leaf_scopes = {n for n in all_nodes if not children_dict.get(n)}

        # --- Phase 2: Topological walk — propagate demand from leaves upward ---
        used_dict: Dict[str, Dict[str, Set[str]]] = {}
        demand_per_scope: Dict[str, Set[str]] = {}  # track demand each scope receives
        visited: Set[str] = set()
        queue = deque(list(leaf_scopes))

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Enqueue parents whose children are all visited
            for upstream in parent_dict.get(current, []):
                if upstream not in visited and all(
                    child in visited for child in children_dict[upstream]
                ):
                    queue.append(upstream)

            sc = scope_name_to_scope.get(current)
            if sc is None:
                # current is a model UID for an external table (no scope, just a node)
                continue

            sc_uid = scope_name_to_uid.get(current, current.split(":", 1)[0])

            if current in leaf_scopes:
                # Leaf: all projections are final output — all demanded
                demand = set(sc.proj_order)
                used_dict[current] = self._collect_local_source_usage_for_demand(
                    sc_uid, sc, demanded_outputs=set(), force_all_proj=True
                )
            else:
                # Aggregate demand from all consumers of this scope
                demand = set()
                force_all = False
                for consumer_sc_name in children_dict.get(current, []):
                    consumer_sc = scope_name_to_scope.get(consumer_sc_name)
                    if consumer_sc and consumer_sc.has_star:
                        force_all = True
                        break
                    if consumer_sc_name in used_dict:
                        demand.update(used_dict[consumer_sc_name].get(current, set()))

                if force_all:
                    demand = set(sc.proj_order)

                used_dict[current] = self._collect_local_source_usage_for_demand(
                    sc_uid, sc, demanded_outputs=demand, force_all_proj=force_all
                )

            demand_per_scope[current] = demand

        # --- Phase 3: Detect prunable JOINs ---
        prunable_joins: List[PruneJoinOpportunity] = []

        for (uid, sc_id, sc_name), sc in self._full_scope_cache.items():
            if not sc.joins:
                continue

            # Build join alias -> JoinInfo
            join_alias_to_info: Dict[str, JoinInfo] = {}
            for ji in sc.joins:
                if ji.source_alias:
                    join_alias_to_info[ji.source_alias] = ji

            if not join_alias_to_info:
                continue

            # Build alias -> upstream scope name
            alias_to_scope_name: Dict[str, str] = {}
            for alias, (skind, sobj) in sc.sources.items():
                if skind in ("subquery", "cte") and isinstance(sobj, Scope):
                    alias_to_scope_name[alias] = sobj.scope_name
                elif skind == "table" and isinstance(sobj, str):
                    resolved_uid = self.art.uid_by_relation.get(sobj)
                    if resolved_uid:
                        alias_to_scope_name[alias] = resolved_uid

            # Compute the demand this scope faces
            demand = demand_per_scope.get(sc_name, set())
            force_all = sc_name in leaf_scopes

            # Get tag-separated usage
            tagged_usage = self._collect_local_source_usage_by_tag(
                uid, sc, demanded_outputs=demand, force_all_proj=force_all
            )

            # Detect columns used in IS NULL / IS NOT NULL checks (non-substitutable)
            null_checked_by_alias = self._collect_null_checked_source_cols(sc)

            # Check each joined source
            for alias, ji in join_alias_to_info.items():
                upstream_scope_name = alias_to_scope_name.get(alias)
                if not upstream_scope_name:
                    continue

                # Columns from this source used in non-JOIN_ON contexts
                proj_cols = tagged_usage["PROJ"].get(upstream_scope_name, set())
                filter_cols = tagged_usage["FILTER"].get(upstream_scope_name, set())
                group_cols = tagged_usage["GROUP_BY"].get(upstream_scope_name, set())
                join_on_cols = tagged_usage["JOIN_ON"].get(upstream_scope_name, set())
                non_join_usage = proj_cols | filter_cols | group_cols

                substitutions: Dict[str, str] = {}

                if non_join_usage:
                    # Check if all non-join usage is substitutable via equi-join equivalences
                    equi_pairs = self._extract_equijoin_pairs(ji.on_expr)

                    # Build substitution map: for this alias, col -> (other_alias, other_col)
                    equiv_map: Dict[str, Tuple[str, str]] = {}
                    for (a_alias, a_col), (b_alias, b_col) in equi_pairs:
                        if a_alias == alias:
                            equiv_map[a_col.lower()] = (b_alias, b_col)
                        elif b_alias == alias:
                            equiv_map[b_col.lower()] = (a_alias, a_col)

                    # Check if every column in non_join_usage has an equivalent
                    # Columns used in IS NULL / IS NOT NULL are NOT substitutable
                    null_checked_cols = null_checked_by_alias.get(alias, set())
                    all_substitutable = True
                    for col in non_join_usage:
                        if col.lower() in null_checked_cols:
                            all_substitutable = False
                            break
                        equiv = equiv_map.get(col.lower())
                        if equiv:
                            other_alias, other_col = equiv
                            substitutions[f"{alias}.{col}"] = f"{other_alias}.{other_col}"
                        else:
                            all_substitutable = False
                            break

                    if not all_substitutable:
                        continue  # this join source has real usage, not prunable

                # At this point: either non_join_usage is empty (direct prune)
                # or all non_join_usage is substitutable (substitution prune)
                confidence, reason = self._classify_join_pruneability(ji.jtype, sc.has_star)

                if substitutions:
                    reason += " Requires column substitution in downstream references."

                # Get column count for impact estimation
                exposed = self._source_exposed_cols(uid, sc, alias)
                col_count = len(exposed) if exposed else None

                # Determine the source reference
                skind, sobj = sc.sources[alias]
                if skind in ("subquery", "cte") and isinstance(sobj, Scope):
                    source_ref = sobj.scope_name
                elif skind == "table" and isinstance(sobj, str):
                    source_ref = sobj
                else:
                    source_ref = alias

                prunable_joins.append(PruneJoinOpportunity(
                    model_uid=uid,
                    scope_name=sc_name,
                    join_type=ji.jtype,
                    unused_source_alias=alias,
                    unused_source_ref=source_ref,
                    unused_source_kind=skind,
                    confidence=confidence,
                    reason=reason,
                    unused_column_count=col_count,
                    join_on_columns_from_unused=join_on_cols,
                    substitutions_needed=substitutions,
                ))

        # --- Phase 4: Detect unused projections ---
        unused_projections: List[UnusedProjectionOpportunity] = []

        for (uid, sc_id, sc_name), sc in self._full_scope_cache.items():
            if sc_name in leaf_scopes:
                continue  # leaf projections are final output

            demand = demand_per_scope.get(sc_name)
            if demand is None:
                continue  # not visited (orphan scope)

            for col_name in sc.proj_order:
                if col_name not in demand:
                    unused_projections.append(UnusedProjectionOpportunity(
                        model_uid=uid,
                        scope_name=sc_name,
                        column_name=col_name,
                        total_projected=len(sc.proj_order),
                    ))

        return PruneAnalysisResult(
            prunable_joins=prunable_joins,
            unused_projections=unused_projections,
            models_analyzed=len(all_uids),
            scopes_analyzed=len(self._full_scope_cache),
        )




































    # compare dataflows across models and find subsets with common patterns
    def group_models_with_same_dataflows(self, unique_ids: List[str]):
        uid_df_set = defaultdict(set)
        uid_col_sets = defaultdict(set)
        
        for uid in unique_ids:
            scope = self._get_or_build_scope(uid)
            selected_columns = list(scope.proj_order)
            for col in selected_columns:
                expr = self._trace_projection_expr(uid=uid, scope=scope, proj_name=col, seen=set())

                uid_col_sets[uid].add(col)
                # if expr is DFLiteral, skip
                if isinstance(expr, DFLiteral):
                    # print(f"Skipping literal column for uid: {uid}, col: {col}")
                    continue
                
                # print(f"Column {col}: {expr}")
                sig = fingerprint_dfexpr(expr)
                # print(f"  Fingerprint: {sig}")
                uid_df_set[uid].add(sig)
                
                
                
            # Only keep uids with non-empty sets
        non_empty_ids = [uid for uid in unique_ids if uid_df_set[uid]]

        # 2. Build an undirected graph based on subset relationships
        graph: Dict[str, Set[str]] = defaultdict(set)

        for i in range(len(non_empty_ids)):
            uid_i = non_empty_ids[i]

            for j in range(i + 1, len(non_empty_ids)):
                uid_j = non_empty_ids[j]

                if uid_df_set[uid_i] == uid_df_set[uid_j] and uid_col_sets[uid_i] == uid_col_sets[uid_j]:
                    graph[uid_i].add(uid_j)
                    graph[uid_j].add(uid_i)

        # 3. Compute connected components with DFS/BFS
        components: List[List[str]] = []
        visited: Set[str] = set()

        for uid in non_empty_ids:
            if uid in visited:
                continue

            stack = [uid]
            visited.add(uid)
            comp = []

            while stack:
                cur = stack.pop()
                comp.append(cur)

                for nbr in graph[cur]:
                    if nbr not in visited:
                        visited.add(nbr)
                        stack.append(nbr)

            components.append(comp)

        # Optionally: print components with size > 1
        for comp in components:
            if len(comp) > 1:
                print(f"Component ({len(comp)} models):")
                for uid in comp:
                    print(f" - {uid}")
                    
        # save the results to a file so I can reuse later
        # with open("outputs/dataflow_connected_components.json", "w") as f:
        #     out = {
        #         "components": components,
        #     }
        #     json.dump(out, f, indent=2)
        #     print("Wrote: outputs/dataflow_connected_components.json")

        return components
    
    def group_models_with_more_fine_grained_dataflows(self, unique_ids: List[str]):
        uid_col_df_set = defaultdict(set)
        uid_pred_df_set = defaultdict(set)
        uid_join_df_set = defaultdict(set)
        uid_col_name_sets = defaultdict(set)
        
        for uid in unique_ids:
            scope = self._get_or_build_scope(uid)
            if scope is None:
                # print(f"Warning: no scope for uid: {uid}")
                continue
            
            selected_columns = list(scope.proj_order)
            uid_col_name_sets[uid] = set(selected_columns)
            
            for col in selected_columns:
                expr = self._trace_projection_expr(uid=uid, scope=scope, proj_name=col, seen=set())
                # if expr is DFLiteral, skip
                if isinstance(expr, DFLiteral):
                    continue

                sig = fingerprint_dfexpr(expr)
                uid_col_df_set[uid].add(sig)
            
            pred_dfs = self.extract_predicate_exprs(uid)
            join_dfs = self.extract_join_exprs(uid)
            
            # print(f"Processing predicates and joins for uid: {uid}")
            # print(f"size of joins: {len(join_dfs)}")
            # for j in join_dfs:
            #     print(f"  Join type: {j.jtype}, expr: {j.expr}")
            
            for pred in pred_dfs:
                sig = fingerprint_dfexpr(pred.expr)
                uid_pred_df_set[uid].add(sig)
            for join in join_dfs:
                sig = fingerprint_dfexpr(join.expr)
                uid_join_df_set[uid].add(sig)
        
        # print number of elements in _uid_scope_cache
        
        # print(f"Number of cached scopes: {len(self._uid_scope_cache)}")
        # for uid, scope in self._uid_scope_cache.items():
        #     print(f"  Cached scope uid: {uid}, scope id: {scope.scope_id}, scope name: {scope.scope_name}")
        
                
        # Only keep uids with non-empty sets
        non_empty_ids = [uid for uid in unique_ids if uid_col_df_set[uid]]

        # 2. Build an undirected graph based on subset relationships
        graph: Dict[str, Set[str]] = defaultdict(set)

        for i in range(len(non_empty_ids)):
            uid_i = non_empty_ids[i]

            for j in range(i + 1, len(non_empty_ids)):
                uid_j = non_empty_ids[j]
                
                if len(uid_col_df_set[uid_i]) != len(uid_col_df_set[uid_j]) or \
                     len(uid_pred_df_set[uid_i]) != len(uid_pred_df_set[uid_j]) or \
                        len(uid_join_df_set[uid_i]) != len(uid_join_df_set[uid_j]):
                    # print(f"Skipping comparison between {uid_i} and {uid_j} due to different set sizes.")
                    # print(f"  uid_i sizes: cols={len(uid_col_df_set[uid_i])}, preds={len(uid_pred_df_set[uid_i])}, joins={len(uid_join_df_set[uid_i])}")
                    # print(f"  uid_j sizes: cols={len(uid_col_df_set[uid_j])}, preds={len(uid_pred_df_set[uid_j])}, joins={len(uid_join_df_set[uid_j])}")
                    continue
                
                col_edit_dis = set_edit_distance_substitution(uid_col_df_set[uid_i], uid_col_df_set[uid_j])
                pred_edit_dis = set_edit_distance_substitution(uid_pred_df_set[uid_i], uid_pred_df_set[uid_j])
                join_edit_dis = set_edit_distance_substitution(uid_join_df_set[uid_i], uid_join_df_set[uid_j])
                
                # print(f"Comparing {uid_i} and {uid_j}: col_edit_dis={col_edit_dis}, pred_edit_dis={pred_edit_dis}, join_edit_dis={join_edit_dis}")

                if uid_col_name_sets[uid_i] == uid_col_name_sets[uid_j] and col_edit_dis == 0 and pred_edit_dis <= 1 and join_edit_dis == 0:
                    graph[uid_i].add(uid_j)
                    graph[uid_j].add(uid_i)

        # 3. Compute connected components with DFS/BFS
        components: List[List[str]] = []
        visited: Set[str] = set()

        for uid in non_empty_ids:
            if uid in visited:
                continue

            stack = [uid]
            visited.add(uid)
            comp = []

            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nbr in graph[cur]:
                    if nbr not in visited:
                        visited.add(nbr)
                        stack.append(nbr)

            components.append(comp)
        return components

    
    
        

    def extract_column_exprs(self, unique_id: str, selected_columns: Optional[List[str]] = None) -> List[ColumnDF]:
        node = self.art.nodes_by_uid[unique_id]
        sql = node.compiled_sql or node.raw_sql
        if not sql:
            # Root node without SQL: each requested column is a DFRef leaf
            cols = selected_columns or []
            return [
                ColumnDF(
                    target_scope_id=-1,
                    target_model=unique_id,
                    target_column=col,
                    expr=self._root_ref(node, col),
                )
                for col in cols
            ]

        scope = self._get_or_build_scope(unique_id)

        if selected_columns is None:
            selected_columns = list(scope.proj_order)

        out: List[ColumnDF] = []
        for col in selected_columns:
            expr = self._trace_projection_expr(uid=unique_id, scope=scope, proj_name=col, seen=set())
            out.append(ColumnDF(
                target_scope_id=scope.scope_id,
                target_model=unique_id,
                target_column=col,
                expr=expr,
            ))
        return out



    '''
    def extract_predicate_exprs(self, unique_id: str) -> List[PredicateDF]:
        node = self.art.nodes_by_uid[unique_id]
        sql = node.compiled_sql or node.raw_sql
        if not sql:
            return []
        
        scope = self._get_or_build_scope(unique_id)

        preds: List[PredicateDF] = []
        visited_scopes: Set[int] = set()
        
        for sc in self._walk_scopes_within_model(scope):
            
            if sc.scope_id in visited_scopes:
                continue
            visited_scopes.add(sc.scope_id)
            
            for tag, pred in sc.filters:
                df_bool = self._expr_to_df(unique_id, sc, pred, seen=set())
                preds.append(PredicateDF(scope_id=sc.scope_id, scope_name=sc.scope_name, tag=tag, expr=df_bool))
        return preds

    def extract_join_exprs(self, unique_id: str) -> List[JoinDF]:
        node = self.art.nodes_by_uid[unique_id]
        sql = node.compiled_sql or node.raw_sql
        if not sql:
            return []

        scope = self._get_or_build_scope(unique_id)
        
        # print("Extracting join expressions...")
        # print(f"Model uid: {unique_id}, scope id: {scope.scope_id}, scope name: {scope.scope_name}")
        

        joins_out: List[JoinDF] = []
        
        visited_scopes: Set[int] = set()
        
        for sc in self._walk_scopes_within_model(scope):
            if sc.scope_id in visited_scopes:
                continue
            visited_scopes.add(sc.scope_id)
            
            
            # TODO: use memoization to avoid re-processing scopes
            
            # print scope name and scope id
            # print(f"Processing scope: {sc.scope_name}, id: {sc.scope_id}")
            
            for j in sc.joins:
                if j.on_expr is None:
                    continue
                df_bool = self._expr_to_df(unique_id, sc, j.on_expr, seen=set())
                
                # print(df_bool)
                
                joins_out.append(JoinDF(
                    scope_id=sc.scope_id,
                    scope_name=sc.scope_name,
                    jtype=j.jtype,
                    expr=df_bool
                ))
        
        # print("Total visited scopes for joins:")
        # print(len(visited_scopes))
                
        return joins_out

    '''


























    # ---------------- Helpers: scope traversal ----------------
    # xxxxx

    # Walk scopes in this model, starting from the root scope. Yields each scope once.
    def _walk_scopes_within_model(self, scope: Scope) -> Iterable[Scope]:
        stack = [scope]
        while stack:
            sc = stack.pop()
            yield sc
            
            for _, (stype, target) in sc.sources.items():
                if stype in ("subquery", "cte"):
                    stack.append(target)
            for child in sc.set_children:
                stack.append(child)


    # Walk scopes with depth (relative to the root scope). Yields (scope, depth) pairs. Depth=0 for the root scope, +1 for direct children, etc.
    # Walk scopes beyond the current model
    def _walk_scopes_across_model_with_depth(self, root: Scope):

        seen = set()
        stack = [(root, 0)]
        while stack:
            sc, d = stack.pop()
            if sc.scope_id in seen:
                continue
            seen.add(sc.scope_id)
            yield sc, d
            sources = sc.sources or {}
            for _, (stype, target) in sources.items():
                if stype in ("subquery", "cte"):
                    stack.append((target, d + 1))
                else:
                    this_uid = self.art.uid_by_relation.get(target)
                    if this_uid is not None:
                        if this_uid in self._uid_scope_cache:
                            this_scope = self._uid_scope_cache[this_uid]
                            stack.append((this_scope, d + 1))
                            
            for child in sc.set_children:
                stack.append((child, d + 1))

    # ---------------- Core: column tracing to DFExpr ----------------

    def _trace_projection_expr(
        self,
        uid: str,
        scope: Scope,
        proj_name: str,
        seen: Set[Tuple[str, str]],
    ) -> DFExpr:
        memo_key = (uid, scope.scope_id, proj_name)
        if memo_key in self._memo_col:
            return self._memo_col[memo_key]

        # Set-op: combine children by position
        if scope.set_op:
            try:
                idx = scope.proj_order.index(proj_name)
            except ValueError:
                # Column not present; emit identity ref to mark unresolved at this scope
                out = DFRef(scope_id=scope.scope_id, column=proj_name, model_uid=uid)
                self._memo_col[memo_key] = out
                return out

            child_exprs: List[DFExpr] = []
            for child in scope.set_children:
                child_name = proj_name
                if idx < len(child.proj_order):
                    child_name = child.proj_order[idx]
                child_exprs.append(self._trace_projection_expr(uid, child, child_name, seen))

                
            out = DFTransform(op="set_op", args=child_exprs, attrs={"op": scope.set_op, "distinct": bool(scope.set_distinct)})
            self._memo_col[memo_key] = out
            return out

        expr = scope.projections.get(proj_name)

        # SELECT * passthrough
        if expr is None and scope.has_star:
            # Prefer concrete table hits using catalog
            for alias, (stype, target) in scope.sources.items():
                if stype == "table" and self.relation_has_col(target, proj_name):
                    return self._column_from_table(uid, scope, relation_sql=target, col_name=proj_name, seen=seen)
            # Then sub-scopes
            for alias, (stype, target) in scope.sources.items():
                if stype in ("subquery", "cte"):
                    sub_scope: Scope = target
                    if proj_name in sub_scope.projections or sub_scope.has_star:
                        return self._trace_projection_expr(uid, sub_scope, proj_name, seen)
            # Fallback: if catalog missed, try table sources anyway
            table_sources = [(alias, target) for alias, (stype, target) in scope.sources.items() if stype == "table"]
            if len(table_sources) == 1:
                _, rel = table_sources[0]
                return self._column_from_table(uid, scope, relation_sql=rel, col_name=proj_name, seen=seen)
            elif len(table_sources) > 1:
                for _, rel in table_sources:
                    upstream_uid = self.art.relation_to_uid(sanitize_relation(rel))
                    if upstream_uid is not None:
                        return self._column_from_table(uid, scope, relation_sql=rel, col_name=proj_name, seen=seen)

        if expr is None:
            # Unknown projection at this scope → emit a ref placeholder
            out = DFRef(scope_id=scope.scope_id, column=proj_name, model_uid=uid)
            self._memo_col[memo_key] = out
            return out

        # Convert this projection expression into a DFExpr tree
        out = self._expr_to_df(uid, scope, expr, seen)
        self._memo_col[memo_key] = out
        return out

    def _column_from_table(
        self,
        current_uid: str,
        scope: Scope,
        relation_sql: str,
        col_name: str,
        seen: Set[Tuple[str, str]],
    ) -> DFExpr:
        
        rel_key = sanitize_relation(relation_sql)
        upstream_uid = self.art.relation_to_uid(rel_key)
        if upstream_uid is None:
            # External/unknown root
            return DFRef(scope_id=None, column=col_name, relation=rel_key, model_uid=None)

        key = (upstream_uid, col_name)
        if key in seen:
            # Cycle guard
            print(f"Warning: cycle detected when tracing column {col_name} in model {upstream_uid}")
            return DFRef(scope_id=scope.scope_id, column=col_name, model_uid=current_uid, relation=rel_key)

        seen.add(key)
        try:
            upstream = self.art.nodes_by_uid[upstream_uid]
            if self.art.is_root_node(upstream_uid):
                return DFRef(scope_id=None, column=col_name, relation=upstream.relation, model_uid=upstream_uid)

            sql = upstream.compiled_sql or upstream.raw_sql
            if not sql:
                return DFRef(scope_id=None, column=col_name, relation=upstream.relation, model_uid=upstream_uid)
            
            sub_scope = self._get_or_build_scope(upstream_uid)
            
            return self._trace_projection_expr(upstream_uid, sub_scope, col_name, seen)
        finally:
            seen.remove(key)

    # ---------------- Expression → DFExpr (columns, funcs, arith, boolean) ----------------

    def _expr_to_df(self, uid: str, scope: Scope, e: exp.Expression, seen: Set[Tuple[str, str]]) -> DFExpr:
        # Literals
        if isinstance(e, exp.Literal):
            if e.is_string:
                return DFLiteral(value=e.this, dtype="STRING")
            if e.is_int:
                return DFLiteral(value=int(e.this), dtype="INT")
            if e.is_number:
                try:
                    return DFLiteral(value=float(e.this), dtype="FLOAT")
                except Exception:
                    return DFLiteral(value=e.this, dtype="NUMBER")
            return DFLiteral(value=e.this, dtype=None)

        # Column references (qualified/unqualified)
        if isinstance(e, exp.Column):
            this_ident = e.args.get("this")
            c_name = this_ident.name if isinstance(this_ident, exp.Identifier) else e.sql(self.dialect)
            tbl_ident = e.args.get("table")
            tbl_alias = tbl_ident.name if isinstance(tbl_ident, exp.Identifier) else None

            if tbl_alias and tbl_alias in scope.sources:
                stype, target = scope.sources[tbl_alias]
                if stype == "table":
                    return self._column_from_table(uid, scope, relation_sql=target, col_name=c_name, seen=seen)
                elif stype in ("subquery", "cte"):
                    sub_scope: Scope = target
                    return self._trace_projection_expr(uid, sub_scope, c_name, seen)
            else:
                # Unqualified: try table sources first using catalog
                for alias, (stype, target) in scope.sources.items():
                    if stype == "table" and self.relation_has_col(target, c_name):
                        return self._column_from_table(uid, scope, relation_sql=target, col_name=c_name, seen=seen)
                # Try subquery/cte
                for alias, (stype, target) in scope.sources.items():
                    if stype in ("subquery", "cte"):
                        sub_scope: Scope = target
                        if c_name in sub_scope.projections or sub_scope.has_star:
                            return self._trace_projection_expr(uid, sub_scope, c_name, seen)
                # Fallback: if catalog lookup missed, try table sources anyway
                # (handles cases where catalog is unavailable or incomplete)
                table_sources = [(alias, target) for alias, (stype, target) in scope.sources.items() if stype == "table"]
                if len(table_sources) == 1:
                    # Single table source: unambiguous, resolve directly
                    _, rel = table_sources[0]
                    return self._column_from_table(uid, scope, relation_sql=rel, col_name=c_name, seen=seen)
                elif len(table_sources) > 1:
                    # Multiple table sources: try each via _column_from_table
                    # (upstream model will fail gracefully if column doesn't exist)
                    for _, rel in table_sources:
                        upstream_uid = self.art.relation_to_uid(sanitize_relation(rel))
                        if upstream_uid is not None:
                            return self._column_from_table(uid, scope, relation_sql=rel, col_name=c_name, seen=seen)
                # Fallback: ref to current scope (unresolved binding)
                return DFRef(scope_id=scope.scope_id, column=c_name, model_uid=uid)


        # transforms / functions
        
        
        # ================== works pretty well ==================
        if isinstance(e, exp.Cast):
            val = e.args.get("this")
            to_type = e.args.get("to").sql(self.dialect) if e.args.get("to") else "UNKNOWN"
            v_expr = self._expr_to_df(uid, scope, val, seen.copy())
            return DFTransform(op="cast", args=[v_expr], attrs={"to_type": to_type})
        
        if isinstance(e, exp.Coalesce):
            args = []
            for a in (e.args.get("expressions") or []):
                args.append(self._expr_to_df(uid, scope, a, seen.copy()))
            return DFTransform(op="coalesce", args=args, attrs={})
        
        if isinstance(e, exp.DPipe):
            left  = self._expr_to_df(uid, scope, e.args.get("this"),       seen.copy())
            right = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())

            # Flatten nested concats into a single n-ary node
            def _collect_concat_args(dfexpr, out):
                if isinstance(dfexpr, DFTransform) and dfexpr.op == "concat":
                    for a in dfexpr.args:
                        _collect_concat_args(a, out)
                else:
                    out.append(dfexpr)

            args: List[DFExpr] = []
            _collect_concat_args(left, args)
            _collect_concat_args(right, args)

            return DFTransform(
                op="concat",
                args=args,
                attrs={"operator": "||", "safe": bool(e.args.get("safe"))}
            )
            
        if isinstance(e, exp.AggFunc):
            return self._agg_to_df(uid, scope, e, seen)
        
        # --- Window functions: ROW_NUMBER(), DENSE_RANK(), SUM(x) OVER(...), etc. ---
        if isinstance(e, exp.Window):
            func_node = e.args.get("this")

            # 1) Function name + arguments (if any)
            def _window_func_and_args(f: exp.Expression) -> Tuple[str, List[DFExpr]]:
                # exp.Anonymous(this="dense_rank") or other unknown names
                if isinstance(f, exp.Anonymous):
                    name = (f.args.get("this") or "").lower()
                    fx = [self._expr_to_df(uid, scope, a, seen.copy())
                        for a in (f.args.get("expressions") or [])]
                    return (name or "anonymous", fx)
                # Known functions (RowNumber, Sum, Avg, etc.) derive from exp.Func
                if isinstance(f, exp.Func):
                    fname = f.__class__.__name__.lower()

                    exprs = (f.args.get("expressions") or [])
                    if not exprs and f.args.get("this") is not None:
                        # some Funcs put the primary arg in "this"
                        exprs = [f.args.get("this")]
                    fx = [self._expr_to_df(uid, scope, a, seen.copy()) for a in exprs]
                    return (fname, fx)
                # Rare: something expression-like as "this"
                if isinstance(f, exp.Expression):
                    return ("identity", [self._expr_to_df(uid, scope, f, seen.copy())])

            fname, fargs = _window_func_and_args(func_node)
            df_func = DFTransform(op="func", args=fargs, attrs={"name": fname})

            # 2) PARTITION BY
            part_items = []
            for p in (e.args.get("partition_by") or []):
                part_items.append(self._expr_to_df(uid, scope, p, seen.copy()))

            # 3) ORDER BY (list of Ordered nodes)
            order_items = []
            order = e.args.get("order")
            if isinstance(order, exp.Order):
                for ord_node in (order.args.get("expressions") or []):
                    if not isinstance(ord_node, exp.Ordered):
                        continue
                    oexpr = ord_node.args.get("this")
                    order_items.append({
                        "expr": self._expr_to_df(uid, scope, oexpr, seen.copy()),
                        "desc": bool(ord_node.args.get("desc")),
                        "nulls_first": bool(ord_node.args.get("nulls_first")),
                        "nulls_last": bool(ord_node.args.get("nulls_last")),
                    })

            # 4) Frame/spec (optional; None in your examples)
            spec = e.args.get("spec") or e.args.get("frame")
            def _window_frame_to_attrs(spec_node: Optional[exp.Expression]) -> Optional[Dict[str, Any]]:
                if not spec_node:
                    return None
                try:
                    kind  = spec_node.args.get("kind")
                    start = spec_node.args.get("start")
                    end   = spec_node.args.get("end")
                    return {
                        "kind":  kind.sql(self.dialect)  if isinstance(kind,  exp.Expression) else (str(kind)  if kind  is not None else None),
                        "start": start.sql(self.dialect) if isinstance(start, exp.Expression) else (str(start) if start is not None else None),
                        "end":   end.sql(self.dialect)   if isinstance(end,   exp.Expression) else (str(end)   if end   is not None else None),
                    }
                except Exception:
                    # Best-effort fallback
                    return {"sql": spec_node.sql(self.dialect)}

            frame_attrs = _window_frame_to_attrs(spec)

            return DFTransform(
                op="window",
                args=[df_func],
                attrs={
                    "partition_by": part_items,
                    "order_by": order_items,
                    "frame": frame_attrs,
                },
            )


        # CASE
        if isinstance(e, exp.Case):
            branches = []
            for w in (e.args.get("ifs") or []):
                cond = w.args.get("this")
                then = w.args.get("true")
                branches.append((self._expr_to_df(uid, scope, cond, seen.copy()), self._expr_to_df(uid, scope, then, seen.copy())))
            else_expr = e.args.get("default")
            else_df = self._expr_to_df(uid, scope, else_expr, seen.copy()) if isinstance(else_expr, exp.Expression) else None
            return DFTransform(op="case", args=[], attrs={"branches": branches, "else": else_df})
        

        # Arithmetic
        if isinstance(e, (exp.Add, exp.Sub, exp.Mul, exp.Div)):
            op_map = {exp.Add: "add", exp.Sub: "sub", exp.Mul: "mul", exp.Div: "div"}
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            right = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())
            return DFTransform(op=op_map[type(e)], args=[left, right])

        # Boolean / Comparisons / Membership / Pattern
        if isinstance(e, (exp.And, exp.Or)):
            op = "and" if isinstance(e, exp.And) else "or"
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            right = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())
            return DFTransform(op=op, args=[left, right])
        if isinstance(e, exp.Not):
            return DFTransform(op="not", args=[self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())])
        if isinstance(e, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
            cmp_map = {exp.EQ:"eq", exp.NEQ:"ne", exp.GT:"gt", exp.GTE:"gte", exp.LT:"lt", exp.LTE:"lte"}
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            right = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())
            return DFTransform(op=cmp_map[type(e)], args=[left, right])
        if isinstance(e, exp.Is):
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            right = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())
            return DFTransform(op="is", args=[left, right])
        if isinstance(e, exp.In):
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            options = [self._expr_to_df(uid, scope, x, seen.copy()) for x in (e.args.get("expressions") or [])]
            return DFTransform(op="in", args=[left] + options)
        if isinstance(e, exp.Like):
            left = self._expr_to_df(uid, scope, e.args.get("this"), seen.copy())
            pattern = self._expr_to_df(uid, scope, e.args.get("expression"), seen.copy())
            return DFTransform(op="like", args=[left, pattern])
        
        if isinstance(e, exp.Concat):
            exprs = e.args.get("expressions") or []
            return DFTransform(
                op="concat",
                args=[self._expr_to_df(uid, scope, a, seen.copy()) for a in exprs],
                attrs={
                    "safe": bool(e.args.get("safe")),
                    "coalesce": bool(e.args.get("coalesce")),
                },
            )
        
        # print(f"e.dump: {e.dump()}")
        # print(f"Function: {e.args}")
        # print(e)
        # print("=================================")

        # Default: wrap unknown as function-ish identity of its SQL (keeps info but won't break)
        return DFTransform(op="unknown", args=[], attrs={"sql": e.sql(self.dialect)})


    def _agg_to_df(self, uid: str, scope: Scope, a: exp.AggFunc, seen: Set[Tuple[str, str]]) -> DFTransform:
        # name / distinct
        fname = a.key.lower() if hasattr(a, "key") and a.key else (a.name.lower() if hasattr(a, "name") and a.name else "agg")
        distinct = bool(a.args.get("distinct")) if a.args.get("distinct") is not None else False

        # args:
        args: List[DFExpr] = []
        exprs = (a.args.get("expressions") or [])
        if exprs:
            args = [self._expr_to_df(uid, scope, x, seen.copy()) for x in exprs]
        elif a.args.get("this") is not None:
            # e.g., COUNT(col)
            args = [self._expr_to_df(uid, scope, a.args.get("this"), seen.copy())]
        else:
            # COUNT(*) → star
            pass

        # COUNT(*) special-case
        star = getattr(a, "is_star", False) or bool(a.args.get("star") if hasattr(a.args, "get") else False)

        # FILTER was already merged in _expr_to_df(Filter), but some parsers embed it on AggFunc
        filt = a.args.get("filter")
        filter_df = self._expr_to_df(uid, scope, filt, seen.copy()) if isinstance(filt, exp.Expression) else None

        # group keys from scope
        gkeys = self._group_keys_df(uid, scope)

        attrs = {
            "func": fname,
            "distinct": distinct,
            "group_keys": gkeys,
        }
        if filter_df is not None:
            attrs["filter"] = filter_df
        if star:
            attrs["star"] = True

        # MIN/MAX/etc. may have zero args (e.g., COUNT(*)). Keep args as-is.
        return DFTransform(op="agg", args=args, attrs=attrs)


    # ---------------- Root reference helper ----------------

    def _root_ref(self, node: DbtNode, col: str) -> DFRef:
        return DFRef(scope_id=None, column=col, relation=node.relation, model_uid=node.unique_id)
