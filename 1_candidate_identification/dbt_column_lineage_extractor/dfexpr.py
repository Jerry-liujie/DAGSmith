from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# Dataflow (expression-based, pure transforms)
# ============================================================

class DFExpr:
    """Base class for dataflow expressions."""

@dataclass
class DFRef(DFExpr):
    """Reference to a column defined by a scope (derived/projection) or an external/root relation."""
    scope_id: Optional[int]             # scope producing this column (None for external root)
    column: str
    relation: Optional[str] = None      # filled for roots/external
    model_uid: Optional[str] = None     # dbt unique_id when known

@dataclass
class DFLiteral(DFExpr):
    value: Any
    dtype: Optional[str] = None

@dataclass
class DFTransform(DFExpr):
    """
    A transform node with operator and argument expressions.
    Examples:
      op="identity"          args=[child]
      op="add"/"sub"/...     args=[left, right]
      op="func"              args=[arg1, arg2, ...], attrs={"name": "coalesce"} or other func name
      op="case"              attrs={"branches":[(cond, val), ...], "else": else_expr}
      op="window"            attrs={"spec_sql": "..."} around args=[value_expr]
      op="set_op"            attrs={"op":"UNION/INTERSECT/EXCEPT", "distinct":True/False}, args=[child0_expr, child1_expr]
      op="and"/"or"/"not"    boolean composition
      op="eq"/"lt"/...       comparison
      op="in"/"like"/...     membership/pattern
    """
    op: str
    args: List[DFExpr] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    
    
# Column outputs & predicate/join outputs
@dataclass
class ColumnDF:
    target_scope_id: int
    target_model: str
    target_column: str
    expr: DFExpr

@dataclass
class PredicateDF:
    scope_id: int
    scope_name: str
    tag: str                 # "WHERE" | "HAVING" | "QUALIFY"
    expr: DFExpr             # boolean DFExpr

@dataclass
class JoinDF:
    scope_id: int
    scope_name: str
    jtype: str
    expr: DFExpr             # boolean DFExpr for ON clause

