from __future__ import annotations
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any, Iterable
from .utils import *

# ============================================================
# dbt artifacts (manifest + optional catalog)
# ============================================================

@dataclass
class DbtNode:
    unique_id: str
    name: str
    resource_type: str        # "model" | "seed" | "snapshot" | "source"
    compiled_sql: Optional[str]
    raw_sql: Optional[str]
    database: Optional[str]
    schema_: Optional[str]
    alias: Optional[str]
    relation: str             # sanitized db.schema.name (or source relation)
    depends_on_nodes: List[str]
    materialized: str

def _strip_database(rel: str) -> Optional[str]:
    """Strip the database (first) component from a sanitized relation like db.schema.table -> schema.table."""
    parts = rel.split(".")
    if len(parts) >= 3:
        return ".".join(parts[1:])
    return None

class DbtArtifacts:
    def __init__(self, manifest_path: str, catalog_path: Optional[str] = None):
        self.manifest_path = manifest_path
        self.catalog_path = catalog_path

        self.nodes_by_uid: Dict[str, DbtNode] = {}
        # relation mapping (full and short; case-insensitive assistance)
        self.uid_by_relation: Dict[str, str] = {}
        self.uid_by_relation_lc: Dict[str, str] = {}

        # catalog-powered metadata
        self.relation_columns: Dict[str, Set[str]] = {}        # rel (sanitized) -> {col_lower}
        self.relation_columns_lc: Dict[str, Set[str]] = {}     # rel.lc key
        self.relation_coltypes: Dict[str, Dict[str, str]] = {} # rel -> {col_lower: type}
        self.relation_stats: Dict[str, Dict[str, Any]] = {}    # rel -> adapter stats
        # secondary index: schema.table (no database) -> same data; for cross-env catalog matching
        self._relation_columns_no_db: Dict[str, Set[str]] = {}
        self._relation_coltypes_no_db: Dict[str, Dict[str, str]] = {}

    def load(self) -> None:
        m = json.loads(pathlib.Path(self.manifest_path).read_text())
        
        nodes = m.get("nodes")
        sources = m.get("sources")
        nodes |= sources
        
        for uid, n in nodes.items():
            rt = n.get("resource_type")
            if rt not in {"model", "seed", "snapshot", "source"}:
                continue
            
            if rt == "source":
                rel = sanitize_relation(id_for([n.get("database"), n.get("schema"), n.get("name")]))
            else:
                # sometimes name and alias may be the same, but they can also be different
                # for example, model.the_tuva_project.service_category__inpatient_psychiatric_institutional
                # name: service_category__inpatient_psychiatric_institutional
                # alias: _int_inpatient_psychiatric_institutional
                rel = sanitize_relation(id_for([n.get("database"), n.get("schema"), n.get("alias") or n.get("name")]))

            node = DbtNode(
                unique_id=uid,
                name=n.get("name"),
                resource_type=rt,
                compiled_sql=n.get("compiled_code"),
                raw_sql=n.get("raw_code"),
                database=n.get("database"),
                schema_=n.get("schema"),
                alias=n.get("alias"),
                relation=rel,
                depends_on_nodes=(n.get("depends_on") or {}).get("nodes", []),
                materialized=(n.get("config") or {}).get("materialized", "unknown"),
            )
            self.nodes_by_uid[uid] = node
            if rel:
                self.uid_by_relation[rel] = uid
                self.uid_by_relation_lc[rel.lower()] = uid

        # Catalog (optional)
        if self.catalog_path:
            try:
                cat = json.loads(pathlib.Path(self.catalog_path).read_text())
                nodes = cat.get("nodes") or {}
                sources = cat.get("sources") or {}
                nodes |= sources
                
                for uid, node, in nodes.items():
                    meta = node.get("metadata") or {}
                    rel = sanitize_relation(id_for([meta.get("database"), meta.get("schema"), meta.get("name")]))
                    if not rel:
                        continue
                    # columns
                    cols = {}
                    for c in (node.get("columns") or {}).values():
                        nm = (c.get("name") or "").lower()
                        if not nm:
                            continue
                        tpe = c.get("type") or c.get("data_type")
                        cols[nm] = tpe
                    if cols:
                        self.relation_coltypes[rel] = cols
                        self.relation_columns[rel] = set(cols.keys())
                        self.relation_columns_lc[rel.lower()] = set(cols.keys())
                        # secondary index: strip database prefix for cross-env matching
                        no_db = _strip_database(rel)
                        if no_db:
                            self._relation_columns_no_db[no_db.lower()] = set(cols.keys())
                            self._relation_coltypes_no_db[no_db.lower()] = cols
                        
                    # stats (adapter-dependent)
                    stats = node.get("stats") or {}
                    if stats:
                        self.relation_stats[rel] = stats
            except Exception:
                # Catalog is optional; ignore if unreadable
                pass


    def relation_to_uid(self, relation_sql: str) -> Optional[str]:
        key = sanitize_relation(relation_sql)
        if key in self.uid_by_relation:
            return self.uid_by_relation[key]
        lckey = key.lower()
        if lckey in self.uid_by_relation_lc:
            return self.uid_by_relation_lc[lckey]
        return None

    def is_root_node(self, uid: str) -> bool:
        n = self.nodes_by_uid[uid]
        if n.resource_type in {"seed", "source"}:
            return True
        upstream_models = [x for x in n.depends_on_nodes]
        return len(upstream_models) == 0
    
    def is_leaf_node(self, uid: str) -> bool:
        # n = self.nodes_by_uid[uid]
        # if n.resource_type in {"model", "snapshot"}:
        #     return True
        downstream_models = [x for x in self.nodes_by_uid.values() if uid in x.depends_on_nodes]
        return len(downstream_models) == 0
