"""diagnostic_exec.py

BigQuery dry-run + execute harness for data-invariant diagnostics.

Used by refactor_slot_aware.py's Phase 1.75 stage to verify that opportunities
approved by the critic actually hold against live data. Every diagnostic is a
SELECT that should return ZERO rows iff the invariant the LLM asserted is true.

Flow per diagnostic SQL:
  1. Render `{{ ref('X') }}` → `project.dataset.table_name` using the baseline
     dbt manifest (handles custom aliases and custom schemas from dbt config).
  2. Dry-run first; if `total_bytes_processed` exceeds `bytes_cap`, skip.
  3. Execute with a timeout. Zero rows = passed. Non-zero = failed (capture
     first 5 sample rows).
  4. On any exception = errored. Caller decides gating policy.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.cloud import bigquery


# ---------------------------------------------------------------------------
# Manifest loading — map short dbt model name to fully-qualified BQ name
# ---------------------------------------------------------------------------

def load_short_to_fqn(manifest_path: str | Path) -> Dict[str, str]:
    """Parse a dbt manifest and return {short_name: 'db.schema.alias'}.

    Uses `nodes[uid].alias` (or `name` if alias absent) as the key, and
    `database.schema.alias` as the value. Matches dbt's own `{{ ref() }}`
    resolution.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found: {p}")
    with p.open() as f:
        manifest = json.load(f)

    out: Dict[str, str] = {}
    for _uid, node in manifest.get("nodes", {}).items():
        if node.get("resource_type") != "model":
            continue
        db = node.get("database") or ""
        sch = node.get("schema") or ""
        alias = node.get("alias") or node.get("name") or ""
        short = node.get("name") or alias
        if not (db and sch and alias and short):
            continue
        # Canonical: short_name -> `db.schema.alias` (backticked later at render time)
        out.setdefault(short, f"{db}.{sch}.{alias}")
    return out


# ---------------------------------------------------------------------------
# Ref rendering
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")


def render_refs(sql: str, short_to_fqn: Dict[str, str]) -> tuple[str, list[str]]:
    r"""Substitute `{{ ref('X') }}` with `\`db.schema.alias\`` and return
    (rendered_sql, unresolved_refs). If any ref can't be resolved, it remains
    in the output and is returned in the unresolved list so the caller can
    decide to skip/error.
    """
    unresolved: list[str] = []

    def sub(m: re.Match) -> str:
        short = m.group(1)
        fqn = short_to_fqn.get(short)
        if fqn is None:
            unresolved.append(short)
            return m.group(0)
        return f"`{fqn}`"

    return _REF_RE.sub(sub, sql), unresolved


# ---------------------------------------------------------------------------
# Diagnostic execution
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticResult:
    status: str  # passed | failed | errored | skipped_cost | skipped_unresolved_ref | empty
    rendered_sql: str = ""
    bytes_scanned: Optional[int] = None
    violation_count: Optional[int] = None
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    unresolved_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "rendered_sql": self.rendered_sql,
            "bytes_scanned": self.bytes_scanned,
            "violation_count": self.violation_count,
            "sample_rows": self.sample_rows,
            "error": self.error,
            "unresolved_refs": self.unresolved_refs,
        }


def get_bq_client(project: Optional[str] = None) -> bigquery.Client:
    """Create a BigQuery client using Application Default Credentials.

    Project resolution: explicit arg > env `DBT_BQ_PROJECT` > ADC default.
    """
    proj = project or os.environ.get("DBT_BQ_PROJECT")
    return bigquery.Client(project=proj)


def check_diagnostic(
    diagnostic_sql: str,
    short_to_fqn: Dict[str, str],
    client: bigquery.Client,
    bytes_cap: int = 10 * (1024 ** 3),  # 10 GiB
    timeout_seconds: int = 120,
    sample_row_limit: int = 5,
) -> DiagnosticResult:
    """Render + dry-run + execute a single diagnostic SQL.

    Rules:
      - Empty SQL → `empty` (no-op).
      - Unresolved refs → `skipped_unresolved_ref` (can't run; caller decides).
      - Dry-run bytes > cap → `skipped_cost`.
      - Query returns zero rows → `passed`.
      - Query returns >0 rows → `failed` (sample_rows populated).
      - Any exception → `errored`.
    """
    sql = (diagnostic_sql or "").strip().rstrip(";").strip()
    if not sql:
        return DiagnosticResult(status="empty")

    rendered, unresolved = render_refs(sql, short_to_fqn)
    if unresolved:
        return DiagnosticResult(
            status="skipped_unresolved_ref",
            rendered_sql=rendered,
            unresolved_refs=sorted(set(unresolved)),
        )

    # Dry run
    try:
        dry_cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        dry_job = client.query(rendered, job_config=dry_cfg)
        bytes_scanned = int(dry_job.total_bytes_processed or 0)
    except Exception as ex:
        return DiagnosticResult(
            status="errored",
            rendered_sql=rendered,
            error=f"dry_run: {ex}",
        )

    if bytes_scanned > bytes_cap:
        return DiagnosticResult(
            status="skipped_cost",
            rendered_sql=rendered,
            bytes_scanned=bytes_scanned,
        )

    # Wrap to cap output rows; the diagnostic SHOULD return zero.
    # A LIMIT on top still lets dry-run estimate scan correctly.
    wrapped = f"SELECT * FROM (\n{rendered}\n) AS _diag LIMIT {sample_row_limit}"

    try:
        run_cfg = bigquery.QueryJobConfig(use_query_cache=False)
        job = client.query(wrapped, job_config=run_cfg)
        rows = list(job.result(timeout=timeout_seconds))
    except Exception as ex:
        return DiagnosticResult(
            status="errored",
            rendered_sql=rendered,
            bytes_scanned=bytes_scanned,
            error=f"execute: {ex}",
        )

    if not rows:
        return DiagnosticResult(
            status="passed",
            rendered_sql=rendered,
            bytes_scanned=bytes_scanned,
            violation_count=0,
        )

    # Non-zero rows → failed. rows is capped at sample_row_limit; violation_count
    # is ">=len(rows)" since we LIMITed. If caller wants the exact count, they
    # can re-run with COUNT(*).
    sample = [dict(r.items()) for r in rows]
    # Coerce non-JSON-safe types to strings for downstream serialization
    safe_sample = []
    for r in sample:
        safe_sample.append({k: _coerce(v) for k, v in r.items()})
    return DiagnosticResult(
        status="failed",
        rendered_sql=rendered,
        bytes_scanned=bytes_scanned,
        violation_count=len(rows),  # lower bound; >= this many
        sample_rows=safe_sample,
    )


def _coerce(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    # datetimes, bytes, decimals, structs, arrays → stringify
    try:
        return str(v)
    except Exception:
        return repr(v)
