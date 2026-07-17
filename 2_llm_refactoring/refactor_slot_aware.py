import os
import re
import json
import time
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from utility.gpt import OpenAI, GptResponse


# ---------------------------------------------------------------------------
# Shared cost-model block (single source of truth for all three prompts).
# ---------------------------------------------------------------------------

COST_MODEL_BLOCK = """COST MODEL — BIGQUERY CAPACITY / SLOT-BASED PRICING:
This project runs under slot-based (capacity) pricing. For QUERY EXECUTION,
the cost meter is slot time — CPU and memory time spent on sorts, joins,
hash aggregations, window functions, shuffles, and repeated expensive
subqueries.

What this means in practice:

- BYTES SCANNED is NOT the cost meter. But scanning less data still often
  correlates with less slot time — fewer rows to shuffle, hash,
  and sort. Treat bytes as a signal, not a bill.

- READING EXTRA COLUMNS often DOES cost slot time. Wide rows inflate
  shuffle volume, hash-table size, and sort buffers.

- FILTER POSITIONING matters when the filter changes how many rows enter
  an expensive operator or enables
  partition / cluster pruning. It does NOT matter for bytes-scanned
  reasons — the optimizer handles read-side pushdown. Reason about
  positioning in terms of row counts into the next expensive operator.

- MATERIALIZE an intermediate when reuse × recompute cost clearly exceeds
  the write. Materializing a cheap or single-use intermediate can lose —
  the write is not free.
"""


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_output(resp: GptResponse) -> Dict[str, Any]:
    text = resp.content.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise ValueError(f"Could not find JSON object in model output:\n{text[:500]}")
        return json.loads(text[start:end + 1])

# ---------------------------------------------------------------------------
# DAG representation helpers
# ---------------------------------------------------------------------------

def build_dag_text(subgraph_data: dict, ignore_immutable: bool = False) -> str:
    """Build a compact DAG representation from extracted subgraph JSON."""
    sg = subgraph_data["subgraph"]
    edges = sg.get("edges", {})
    immutable = sg.get("has_external_children", {})

    lines = ["DEPENDENCY GRAPH (parent -> children):", ""]
    for parent, children in sorted(edges.items()):
        if children:
            lines.append(f"  {parent} -> {children}")
    lines.append("")

    if not ignore_immutable and immutable:
        lines.append("IMMUTABLE MODELS (have external consumers — output must stay identical, internals may be rewritten):")
        for uid in sorted(immutable.keys()):
            lines.append(f"  - {uid}")
        lines.append("")

    return "\n".join(lines)


def collect_model_sql(subgraph_data: dict) -> Dict[str, str]:
    """Return {uid: raw_code} for all models with SQL."""
    return {
        uid: m["raw_code"]
        for uid, m in subgraph_data["models"].items()
        if m.get("raw_code")
    }


# ---------------------------------------------------------------------------
# Phase 1: Analyze
# ---------------------------------------------------------------------------

def build_analyze_prompt(
    dag_text: str,
    all_sql: Dict[str, str],
    immutable_uids: List[str],
    ignore_immutable: bool = False,
) -> list[dict]:

    immutable_section = ""
    if not ignore_immutable and immutable_uids:
        immutable_section = (
            "\nIMMUTABLE MODELS:\n"
            "The following models have consumers outside this subgraph. You MAY rewrite\n"
            "their internal SQL to an equivalent version, and you MAY reference them\n"
            "from other models. You may NOT change their OUTPUT — column set, types,\n"
            "or rows produced must remain identical to the original.\n"
            f"Immutable: {immutable_uids}\n"
        )

    system_content = (
        "You are an expert dbt/SQL pipeline analyst preparing a subgraph for refactoring.\n"
        "\n"
        "You are given the full SQL for every model in a subgraph plus its dependency graph.\n"
        "Your task is to ANALYZE (not refactor yet) the subgraph and produce a structured report.\n"
        "\n"
        + COST_MODEL_BLOCK +
        "\n"
        "UNDERSTAND EACH MODEL'S BUSINESS LOGIC FIRST:\n"
        "Read each model the way a domain analyst would: what business concept does\n"
        "it compute, what data assumptions does it encode, and what downstream\n"
        "business question does it serve? Then zoom out: what does the subgraph as\n"
        "a whole achieve — what end-to-end business question does it answer, and\n"
        "what is the conceptual data flow from sources to terminal outputs?\n"
        "\n"
        "This semantic, domain-level read is the LLM's fundamental advantage over\n"
        "a query planner. It surfaces rewrites a planner cannot see, for example:\n"
        "  - Two models computing the same business quantity via different SQL\n"
        "    paths can be unified at the semantic layer.\n"
        "  - Business logic that no downstream model actually consumes can be\n"
        "    removed even if its SQL looks productive.\n"
        "Record these as optimization opportunities when you find them.\n"
        "First record your per-model semantic read in `model_semantics`, then roll\n"
        "up to the subgraph-level synthesis in `subgraph_summary` (see OUTPUT\n"
        "FORMAT). When a business-logic insight becomes an\n"
        "optimization opportunity, articulate its MECHANISM (below) in terms of the\n"
        "underlying CPU work eliminated when the redundant, dead, or semantically-\n"
        "equivalent computation is removed.\n"
        "\n"
        "MODEL FAMILY DETECTION:\n"
        "Many dbt projects contain families of models that do nearly identical work on different\n"
        "data subsets. Identify these families. For each family:\n"
        "  - Select ONE REPRESENTATIVE model that captures the family's full SQL pattern.\n"
        "  - Describe what is shared (in `shared_pattern`) and what varies across members\n"
        "    (in `key_differences`).\n"
        "  - For EACH member (including the representative), populate `member_differences`\n"
        "    with the exact distinguishing literals copied verbatim from the SQL.\n"
        "    Read each member's SQL — do NOT guess or abbreviate values from the model name.\n"
        "  - Member variation that fits one canonical pattern + a\n"
        "    `key_differences` description belongs in ONE family.\n"
        "  - If members have STRUCTURAL variants that no single rep can canonically represent\n"
        "    (different join sets, different aggregation grain, additional/missing\n"
        "    transformation steps), SPLIT them into separate families instead. One rep,\n"
        "    one family.\n"
        "  - Models that don't belong to any family are standalone models.\n"
        "\n"
        "OPTIMIZATION OPPORTUNITIES — REDUCE SLOT TIME:\n"
        "Propose any rewrite that concretely reduces CPU work.\n"
        "\n"
        "You have ONLY the SQL and the DAG. You do NOT have row counts, table sizes,\n"
        "runtime frequencies, or query plans. Do NOT invent them.\n"
        "\n"
        "For each opportunity, answer:\n"
        "  1. MECHANISM: Name the CPU-bound computation eliminated \n"
        "     or redundant business logic that goes away. Must be\n"
        "     concrete.\n"
        "  2. NEW-COST CHECK: Does the rewrite introduce a new expensive\n"
        "     computation? If yes, is it smaller than what it replaces? Apply\n"
        "     only if smaller, or if no new expensive computation is introduced.\n"
        "  3. REASONING LEVEL: Classify the reasoning required to DISCOVER this\n"
        "     opportunity (not the SQL transformation, which always has a prior\n"
        "     analogue — the novel part is finding the opportunity):\n"
        "       Level 0: A query optimizer or linter could find this within a\n"
        "                single SQL file.\n"
        "       Level 1: Requires seeing the same structural pattern across\n"
        "                multiple files. A clone detector or AST matcher could\n"
        "                find it.\n"
        "       Level 2: Requires understanding that differently-named,\n"
        "                differently-structured SQL across models computes the\n"
        "                same business concept. Syntactic matching fails.\n"
        "       Level 3: Requires recognizing semantic equivalence across domain\n"
        "                boundaries. No organizing principle groups these models.\n"
        "\n"
        "Level 2-3 opportunities require semantic understanding that existing\n"
        "tools cannot replicate. Identify them when present, but only propose\n"
        "when the mechanism concretely reduces compute — novelty alone is not\n"
        "sufficient.\n"
        "\n"
        "EXCLUDE only opportunities whose primary benefit is readability or\n"
        "maintenance, or whose compute impact you cannot articulate concretely.\n"
        f"{immutable_section}"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "model_semantics": [\n'
        "    {\n"
        '      "uid": <string>,\n'
        '      "business_concept": <string, 1-2 sentences: what this model computes in domain terms>,\n'
        '      "data_assumptions": <string, conditions the logic implicitly depends on about the source data; "" if none significant>,\n'
        '      "consumed_by_purpose": <string, what downstream business question(s) this serves; "no in-subgraph consumer" if a leaf>\n'
        "    }\n"
        "  ],\n"
        '  "subgraph_summary": {\n'
        '    "business_purpose": <string, 1-2 sentences: what business question(s) this subgraph as a whole answers>,\n'
        '    "pipeline_shape": <string, conceptual flow from sources to terminal outputs in domain terms (the data-flow described at business-entity level, not SQL structure)>,\n'
        '    "key_domain_entities": [<string, the core business entities the subgraph operates on>]\n'
        '  },\n'
        '  "model_families": [\n'
        "    {\n"
        '      "family_name": <string>,\n'
        '      "description": <string, what these models do and why they are a family>,\n'
        '      "representative": <uid>,\n'
        '      "other_members": [<uid>, ...],\n'
        '      "shared_pattern": <string, the SQL pattern shared across the family>,\n'
        '      "key_differences": <string, what varies across members>,\n'
        '      "member_differences": [\n'
        '        {"uid": <uid>, "distinguishing_values": {<param_name>: <exact literal from SQL>, ...}}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "standalone_models": [<uid>, ...],\n'
        '  "costly_patterns": [\n'
        '    <string, description of a repeated/expensive computation>\n'
        "  ],\n"
        '  "optimization_opportunities": [\n'
        "    {\n"
        '      "strategy": <string>,\n'
        '      "mechanism": <string, the CPU-bound computation eliminated or shrunk — name it concretely>,\n'
        '      "new_cost_check": <string, any new expensive computation this rewrite introduces and whether it is smaller than what it replaces; "none" if none>,\n'
        '      "reasoning_level": <0|1|2|3, per the reasoning depth classification above>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Do NOT include any text outside the JSON.\n"
    )

    user_content = (
        f"{dag_text}\n"
        f"SQL DEFINITIONS:\n{json.dumps(all_sql, indent=2)}\n\n"
        "Please analyze this subgraph as described."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Phase 1.5: Performance Critic — adversarial review of proposed opportunities.
# Runs pre-refactoring to filter out opportunities that would likely increase
# slot time or introduce correctness risk.
# ---------------------------------------------------------------------------

def build_critic_prompt(
    dag_text: str,
    all_sql: Dict[str, str],
    analysis: Dict[str, Any],
) -> list[dict]:
    """Build prompt for adversarial review of proposed optimization opportunities."""

    opps_json = json.dumps(analysis.get("optimization_opportunities", []), indent=2)

    system_content = (
        "You are a performance engineer reviewing proposed dbt refactoring strategies.\n"
        "Your job is to find genuine slot-time regressions or correctness breaks.\n"
        "You are an adversary — assume the proposer is overconfident and look for what they missed.\n"
        "\n"
        + COST_MODEL_BLOCK +
        "\n"
        "You are given the full SQL of every model in the subgraph plus the proposed strategies.\n"
        "Under slot pricing, only these things are real risks:\n"
        "\n"
        "A. NEW EXPENSIVE COMPUTE: The proposal introduces a compute step that did\n"
        "   not exist before (e.g., new shuffle, new wide-partition window, etc.)\n"
        "   and the step it replaces was cheaper.\n"
        "\n"
        "B. SPILL / MEMORY PRESSURE: Merging workloads pushes a hash or sort past\n"
        "   available memory such that it spills where it previously did not.\n"
        "\n"
        "C. CORRECTNESS: A wrong rewrite has infinite cost.\n"
        "\n"
        "Evaluate every proposal by the same standard.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "reviews": [\n'
        "    {\n"
        '      "strategy_index": <int, 0-based index of the opportunity>,\n'
        '      "strongest_counterargument": <string, the most concrete mechanism for cost increase>,\n'
        '      "verdict": "approve" | "reject" | "conditional",\n'
        '      "condition": <string, if conditional: what MUST be true for this to be net positive; empty if approve/reject>\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "Verdicts:\n"
        '- "approve": You genuinely cannot find a plausible way this increases slot time.\n'
        '- "reject": You found a concrete mechanism by which this likely increases slot time.\n'
        '- "conditional": The change could help OR hurt depending on a specific condition.\n'
        "  Specify that condition clearly so the refactoring phase can enforce it.\n"
        "\n"
        "Be rigorous. If in doubt, use 'conditional' rather than 'approve'.\n"
        "Do NOT include any text outside the JSON.\n"
    )

    user_content = (
        f"{dag_text}\n"
        f"SQL DEFINITIONS:\n{json.dumps(all_sql, indent=2)}\n\n"
        f"PROPOSED OPTIMIZATION OPPORTUNITIES:\n{opps_json}\n\n"
        "Review each opportunity as described."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Phase 2.6: Adversarial correctness critic — separate LLM call that reviews
# original vs refactored SQL for every rewritten leaf/immutable model.
# Generates correctness_diagnostics that are then executed by
# run_post_refactor_diagnostics().
# ---------------------------------------------------------------------------

def build_correctness_critic_prompt(
    subgraph_data: Dict[str, Any],
    refactored_models: List[Dict[str, Any]],
    all_sql: Dict[str, str],
    optimization_opportunities: List[Dict[str, Any]] = (),
    project_plan: Dict[str, Any] = None,
) -> list[dict]:
    """Build the adversarial correctness critic prompt.

    Classifies models as leaf/immutable/internal using the subgraph metadata,
    pairs original vs refactored SQL for every rewritten leaf/immutable, and
    provides new-model SQL for the critic's analysis.
    """
    if project_plan is None:
        project_plan = {}
    # --- Classify models ---
    component_nodes = set(subgraph_data["subgraph"].get("component_nodes", []))
    ext_children = set(subgraph_data["subgraph"].get("has_external_children", {}).keys())
    edges = subgraph_data["subgraph"].get("edges", {})

    immutable_uids = ext_children

    has_component_child: set = set()
    for parent, children in edges.items():
        for child in children:
            if child in component_nodes:
                has_component_child.add(parent)
    leaf_uids = component_nodes - has_component_child

    # --- Build original-vs-refactored pairs for leaf/immutable ---
    leaf_immutable_pairs: Dict[str, Dict[str, str]] = {}
    new_model_sql: Dict[str, str] = {}

    for m in refactored_models:
        uid = m.get("uid", "")
        if m.get("is_new"):
            new_model_sql[uid] = m.get("rewritten_sql", "")
            continue
        if m.get("removed"):
            continue
        if not m.get("rewritten"):
            continue
        if uid in leaf_uids or uid in immutable_uids:
            leaf_immutable_pairs[uid] = {
                "original_sql": all_sql.get(uid, ""),
                "refactored_sql": m.get("rewritten_sql", ""),
            }

    system_content = (
        "You are an adversarial correctness reviewer for REFACTORED dbt SQL.\n"
        "Your job is to find refactorings that silently change the output of leaf\n"
        "or immutable models. Assume the author is overconfident.\n"
        "\n"
        "YOU ARE GIVEN:\n"
        "- A numbered list of adopted optimization opportunities, each with its\n"
        "  strategy, mechanism, and (optionally) a critic_condition.\n"
        "- The refactor's project summary and new shared model definitions.\n"
        "- Original and refactored SQL for every rewritten leaf and immutable model.\n"
        "- SQL text of new shared models (diagnostics must not\n"
        "  reference new models directly since they do not exist in the database yet).\n"
        "\n"
        "THE CENTRAL QUESTION:\n"
        "For each adopted optimization opportunity, does the refactored SQL\n"
        "preserve the exact result set of every affected leaf and immutable model?\n"
        "\n"
        "For each opportunity, reason about its mechanism: what computation was\n"
        "eliminated or restructured? What data invariant must hold for that\n"
        "elimination to be semantically safe?\n"
        "Emit as many diagnostics as needed; a trivial rewrite\n"
        "can have zero. Be comprehensive on genuinely risky rewrites; empty list is\n"
        "fine when the rewrite is purely structural.\n"
        "\n"
        "If an opportunity carries a `critic_condition`, that condition is a\n"
        "specific assumption the performance critic flagged as uncertain —\n"
        "always emit a diagnostic testing it.\n"
        "\n"
        "If you identify a correctness risk not tied to any specific opportunity,\n"
        "use `opportunity_index: null`.\n"
        "\n"
        "DIAGNOSTIC RELEVANCE:\n"
        "A diagnostic should test whether the REFACTORING can produce different\n"
        "output from the original — not whether the original was deterministic.\n"
        "If both original and refactored SQL share the same nondeterministic\n"
        "behavior (e.g., both pick an arbitrary row among ties via ROW_NUMBER\n"
        "or MAX), do NOT emit a diagnostic testing whether ties exist. That is\n"
        "a data-quality check, not a refactoring-correctness check.\n"
        "Only emit diagnostics for invariants the refactoring NEWLY depends on\n"
        "and the original did not.\n"
        "\n"
        "COST GUIDANCE:\n"
        "Each diagnostic will be executed against live BigQuery. Overly broad\n"
        "diagnostics that scan large tables may exceed byte caps and be\n"
        "auto-skipped, reducing your coverage.\n"
        "Write each diagnostic SQL to be as narrow as possible.\n"
        "\n"
        "A data invariant is a concrete claim about what rows live in source\n"
        "tables, NOT about SQL structure, slot-time cost, or intent. Write each\n"
        "`diagnostic_sql` so it returns ZERO rows when the invariant holds, and\n"
        "rows when it is violated.\n"
        "\n"
        "SQL requirements:\n"
        "  - Valid BigQuery.\n"
        "  - Do not reference newly created models (they do not exist in the database yet).\n"
        "  - Use dbt `{{ ref('model_name') }}` for every table reference; the\n"
        "    runner resolves refs to fully-qualified names via the dbt manifest.\n"
        "\n"
        "Do not skip invariants you believe are \"obviously true\". "
        "If an assumption genuinely cannot be\n"
        "expressed as SQL, add `UNCHECKABLE: <reason>` to `caveats` and LEAVE THAT\n"
        "INVARIANT OUT of `correctness_diagnostics`.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "correctness_diagnostics": [\n'
        '    {"opportunity_index": <int or null, 0-based index into the adopted opportunities list; null if general>,\n'
        '     "invariant": <string, one sentence: what must be true about DATA for this opportunity\'s rewrite to remain row-equivalent>,\n'
        '     "diagnostic_sql": <string, a SELECT that returns ZERO rows iff the invariant holds>,\n'
        '     "source_tables": [<string, table/model refs the query reads>],\n'
        '     "caveats": <string, what this diagnostic would NOT catch, or "">}\n'
        "  ]\n"
        "}\n"
        "\n"
        "Do NOT include any text outside the JSON.\n"
    )

    # --- Build user message ---
    user_parts = []

    if optimization_opportunities:
        user_parts.append("ADOPTED OPTIMIZATION OPPORTUNITIES (0-indexed):")
        for i, opp in enumerate(optimization_opportunities):
            opp_display = {
                "index": i,
                "strategy": opp.get("strategy", ""),
                "mechanism": opp.get("mechanism", ""),
                "reasoning_level": opp.get("reasoning_level"),
            }
            if opp.get("critic_condition"):
                opp_display["critic_condition"] = opp["critic_condition"]
            user_parts.append(json.dumps(opp_display, indent=2))
        user_parts.append("")

    summary = project_plan.get("summary", "")
    if summary:
        user_parts.append("REFACTOR PROJECT SUMMARY:")
        user_parts.append(summary)
        user_parts.append("")

    new_shared = project_plan.get("new_shared_models", [])
    if new_shared:
        user_parts.append("NEW SHARED MODELS (from project plan):")
        user_parts.append(json.dumps(new_shared, indent=2))
        user_parts.append("")

    user_parts.append("LEAF / IMMUTABLE MODELS — original vs refactored SQL:")
    user_parts.append(json.dumps(leaf_immutable_pairs, indent=2))
    user_parts.append("")

    if new_model_sql:
        user_parts.append("NEW SHARED MODELS — SQL text (for your analysis only; do NOT reference these in diagnostic SQL):")
        user_parts.append(json.dumps(new_model_sql, indent=2))
        user_parts.append("")

    user_parts.append(
        "For each adopted opportunity above, identify the data invariants that "
        "must hold for the rewrite to preserve correctness. Emit a "
        "correctness_diagnostics entry for each critical invariant, linking it "
        "to the opportunity via opportunity_index. If you find a correctness "
        "risk not tied to any specific opportunity, use opportunity_index: null."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Post-refactor data-invariant execution (Phase 2.7 correctness_diagnostics)
#
# Executes correctness_diagnostics (from the adversarial critic or legacy
# self-diagnostics) against live BigQuery. Any query returning rows rejects
# the refactoring attempt.
# ---------------------------------------------------------------------------

def run_post_refactor_diagnostics(
    diagnostics_or_plan,
    manifest_path: str,
    bytes_cap: int,
    strict_cost: bool,
) -> tuple[bool, list, list]:
    """Execute correctness diagnostics against BigQuery.

    `diagnostics_or_plan` can be either:
      - a list of diagnostic dicts (from the correctness critic), or
      - a dict (project_plan) with a `correctness_diagnostics` key (legacy).

    Returns `(all_passed, results, failure_records)`.
    """
    import diagnostic_exec as _de

    if isinstance(diagnostics_or_plan, list):
        diagnostics = diagnostics_or_plan
    else:
        diagnostics = diagnostics_or_plan.get("correctness_diagnostics", []) or []
    if not diagnostics:
        return True, [], []

    try:
        short_to_fqn = _de.load_short_to_fqn(manifest_path)
    except Exception as e:
        print(f"    ERROR loading manifest {manifest_path}: {e}")
        # Can't run → treat as no gate (no failure). Caller may choose to warn.
        return True, [{"error": str(e), "stage": "manifest_load"}], []

    client = _de.get_bq_client()
    results: list = []
    failures: list = []
    all_passed = True

    for i, diag in enumerate(diagnostics):
        sql = (diag.get("diagnostic_sql") or "").strip()
        invariant = diag.get("invariant", "")
        if not sql:
            results.append({
                "index": i,
                "invariant": invariant,
                "status": "empty",
            })
            continue

        result = _de.check_diagnostic(sql, short_to_fqn, client, bytes_cap=bytes_cap)
        record = {
            "index": i,
            "opportunity_index": diag.get("opportunity_index"),
            "invariant": invariant,
            "diagnostic_sql": sql,
            **result.to_dict(),
        }
        results.append(record)

        status = result.status
        if status in ("passed", "empty"):
            verdict = "PASS"
        elif status == "skipped_cost" and not strict_cost:
            verdict = "PASS (skipped_cost)"
        else:
            verdict = "FAIL"

        opp_idx = diag.get("opportunity_index")
        opp_tag = f" opp[{opp_idx}]" if opp_idx is not None else ""
        inv_short = invariant[:120]
        print(f"    [{verdict}] diag[{i}]{opp_tag} {status}: {inv_short}")

        if verdict.startswith("FAIL"):
            all_passed = False
            failures.append({
                "opportunity_index": diag.get("opportunity_index"),
                "opportunity_strategy": diag.get("opportunity_strategy", ""),
                "invariant": invariant,
                "diagnostic_sql": sql,
                "status": status,
                "violation_count": result.violation_count,
                "sample_rows": result.sample_rows,
                "error": result.error,
            })
            if status == "failed":
                print(f"        violation_count>={result.violation_count}, bytes_scanned={result.bytes_scanned}")

    return all_passed, results, failures


# ---------------------------------------------------------------------------
# Phase 2: Refactor
# ---------------------------------------------------------------------------

def build_refactor_prompt(
    dag_text: str,
    analysis: Dict[str, Any],
    representative_sql: Dict[str, str],
    standalone_sql: Dict[str, str],
    immutable_sql: Dict[str, str],
    ignore_immutable: bool = False,
    previous_plans: Optional[List[Dict[str, Any]]] = None,
) -> list[dict]:

    system_content = (
        "You are an expert dbt and SQL pipeline refactoring assistant.\n"
        "\n"
        "You are given:\n"
        "  - A dependency graph of the subgraph.\n"
        "  - An analysis identifying model families, representatives, and optimization opportunities.\n"
        "  - Full SQL for REPRESENTATIVE models, standalone models, and immutable models.\n"
        "  - For non-representative family members, the analysis describes how they differ from representatives.\n"
        "\n"
        + COST_MODEL_BLOCK +
        "\n"
        "SEMANTICS:\n"
        "For every leaf node in the ORIGINAL DAG (a model with no downstream dependents inside\n"
        "this subgraph), the refactored project must produce the same result.\n"
        "\n"
        "You MAY create new dbt models and MAY mark existing models as removed.\n"
        "For new models, specify materialization ('table' for reused/expensive, 'view' for lightweight).\n"
    )

    if not ignore_immutable:
        system_content += (
            "\nIMMUTABLE MODELS:\n"
            "Models flagged as immutable have consumers outside this subgraph. You MAY\n"
            "rewrite their internal SQL to an equivalent version, and you MAY reference\n"
            "them from other models. You may NOT change their OUTPUT — column set, types,\n"
            "or rows produced must remain identical to the original.\n"
        )

    system_content += (
        "\nOPTIMIZATION PRIORITY:\n"
        "The analysis identifies optimization opportunities at different reasoning\n"
        "levels (0-3). Implement all opportunities that concretely reduce compute,\n"
        "regardless of level. Actively look for Level 2-3 opportunities — these\n"
        "require semantic understanding that no existing tool can replicate. When\n"
        "two opportunities conflict and offer similar compute reduction, prefer\n"
        "the more novel one. But never force a novel rewrite whose compute benefit\n"
        "is marginal. If you introduce a new shared model, verify its grain\n"
        "preserves downstream row counts.\n"
        "\n"
        "CRITIC-REVIEWED OPPORTUNITIES:\n"
        "The optimization opportunities in the analysis have been reviewed by an\n"
        "adversarial performance critic. Some may carry a `critic_condition` field —\n"
        "if present, verify that condition holds in your rewrite. Do not apply\n"
        "opportunities that were not included in the filtered list.\n"
        "\n"
        "FAMILY PROPAGATION:\n"
        "You are refactoring REPRESENTATIVE models. For each change, describe how it applies\n"
        "to the entire family. Non-representative members will be updated in a later step.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "project_plan": {\n'
        '    "summary": <string, 4-8 sentences describing overall strategy>,\n'
        '    "performance_analysis": {\n'
        '      "before": <string, description of current cost structure>,\n'
        '      "after": <string, description of expected cost after refactoring>,\n'
        '      "estimated_reduction": <string, rough estimate of improvement>\n'
        "    },\n"
        '    "has_more_strategies": <boolean>,\n'
        '    "new_shared_models": [\n'
        '      {"uid": <string>, "purpose": <string>, "reused_by": [<uid>], "materialized": <string>,\n'
        '       "grain": <string, the columns that uniquely identify one row in this model>,\n'
        '       "grain_verification": <string, for each consumer: why the row count entering\n'
        '        its first expensive operator (join, window, sort) is unchanged vs. the original>}\n'
        "    ],\n"
        '    "family_propagation": [\n'
        '      {"family_name": <string>, "representative": <uid>,\n'
        '       "change_description": <string, what changed on the representative and how it applies to every family member; the runner will then re-derive each member\'s SQL via an LLM call using this description + the representative\'s original/refactored SQL + upstream ref SQL>}\n'
        "    ],\n"
        '    "notes": [<string>]\n'
        "  },\n"
        '  "models": [\n'
        '    {"uid": <string>, "rewritten": <bool>, "is_new": <bool>, "removed": <bool>,\n'
        '     "materialized": <string or null>, "rewritten_sql": <string>, "description": <string>}\n'
        "  ]\n"
        "}\n"
        "\n"
        "Important rules:\n"
        "- Include every REPRESENTATIVE and STANDALONE model uid in the output.\n"
        "- Do NOT include non-representative family members (they will be propagated later).\n"
        "- For new models: is_new=true, removed=false, materialized required.\n"
        "- For removed models: removed=true, rewritten_sql=\"\".\n"
        "- For unchanged models: rewritten=false, copy original SQL.\n"
        "- Do NOT include any text outside the JSON.\n"
    )

    # Build user message
    user_parts = [dag_text, ""]

    user_parts.append("ANALYSIS (from Phase 1):")
    user_parts.append(json.dumps(analysis, indent=2))
    user_parts.append("")

    user_parts.append("SQL — REPRESENTATIVE MODELS:")
    user_parts.append(json.dumps(representative_sql, indent=2))
    user_parts.append("")

    if standalone_sql:
        user_parts.append("SQL — STANDALONE MODELS:")
        user_parts.append(json.dumps(standalone_sql, indent=2))
        user_parts.append("")

    if immutable_sql and not ignore_immutable:
        user_parts.append("SQL — IMMUTABLE MODELS:")
        user_parts.append(json.dumps(immutable_sql, indent=2))
        user_parts.append("")

    if previous_plans:
        user_parts.append("PREVIOUS ATTEMPTS:")
        for idx, plan in enumerate(previous_plans, 1):
            user_parts.append(f"--- Attempt {idx} ---")
            user_parts.append(f"Summary: {plan.get('summary')}")
            user_parts.append(f"Performance self-estimate: {plan.get('performance_analysis', {})}")
            failures = plan.get("diagnostic_failures", []) or []
            for f in failures:
                inv = (f.get("invariant", "") or "")[:200]
                opp_strat = f.get("opportunity_strategy", "")
                if opp_strat:
                    user_parts.append(f"  Correctness invariant VIOLATED [opp {f.get('opportunity_index')}: {opp_strat[:80]}]: {inv}")
                else:
                    user_parts.append(f"  Correctness invariant VIOLATED: {inv}")
            if plan.get("abandoned_reason"):
                user_parts.append(f"  ABANDONED: {plan.get('abandoned_reason')}")
            user_parts.append("")

        user_parts.append(
            "The previous attempt(s) failed correctness diagnostics against live "
            "BigQuery data. Propose a DISTINCTLY DIFFERENT strategy or restore the "
            "original logic for the affected model(s). If all reasonable strategies "
            "are exhausted, set has_more_strategies=false."
        )
        user_parts.append("")

    user_parts.append("Please provide your refactored SQL definitions.")

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Phase 3: Propagate
# ---------------------------------------------------------------------------

def build_propagate_prompt(
    family_name: str,
    representative_uid: str,
    refactored_representative_sql: str,
    original_representative_sql: str,
    member_uid: str,
    member_sql: str,
    key_differences: str,
    change_description: str,
    upstream_refs_sql: Optional[Dict[str, str]] = None,
) -> list[dict]:

    system_content = (
        "You are applying a refactoring change from a representative model to a family member.\n"
        "\n"
        "You are given:\n"
        "  - The ORIGINAL SQL of the representative model (before refactoring).\n"
        "  - The REFACTORED SQL of the representative model (after refactoring).\n"
        "  - The SQL of every upstream model (pre-existing OR newly introduced) that\n"
        "    the refactored representative or this member's original SQL references\n"
        "    via {{ ref(...) }}. Use these to\n"
        "    pick correct filter literals, column names, and join keys for the member.\n"
        "    Do NOT infer the member's filter literal from its dbt model name when the\n"
        "    upstream SQL shows the real set of emitted values.\n"
        "  - A description of what changed.\n"
        "  - The ORIGINAL SQL of the family member you must update.\n"
        "  - A description of how the family member differs from the representative.\n"
        "\n"
        "Apply the analogous refactoring to the family member. Preserve the member's\n"
        "unique differences (e.g., different filter values, extra columns) while applying\n"
        "the structural changes from the representative.\n"
        "\n"
        "Return ONLY a JSON object:\n"
        '{"uid": <string>, "rewritten_sql": <string>, "description": <string>}\n'
        "\n"
        "Do NOT include any text outside the JSON.\n"
    )

    user_parts = [
        f"Family: {family_name}",
        f"Change description: {change_description}",
        "",
        f"REPRESENTATIVE ({representative_uid}) — ORIGINAL:\n{original_representative_sql}",
        "",
        f"REPRESENTATIVE ({representative_uid}) — REFACTORED:\n{refactored_representative_sql}",
        "",
    ]

    if upstream_refs_sql:
        user_parts.append("UPSTREAM MODELS referenced by the refactored representative and/or this member:")
        for ref_name, ref_sql in upstream_refs_sql.items():
            user_parts.append(f"--- {ref_name} ---")
            user_parts.append(ref_sql)
            user_parts.append("")

    user_parts.extend([
        f"KEY DIFFERENCES of this member from representative:\n{key_differences}",
        "",
        f"FAMILY MEMBER ({member_uid}) — ORIGINAL (apply analogous changes):\n{member_sql}",
        "",
        "Please provide the refactored SQL for this family member.",
    ])

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Per-run file backup helpers
# ---------------------------------------------------------------------------

def _backup_file(path: str) -> Optional[str]:
    """Rename `path` with a timestamp suffix. Returns the new path, or None if it did not exist."""
    if not os.path.exists(path):
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{ts}"
    os.rename(path, backup_path)
    return backup_path


def _backup_group_files(output_dir: str, group_id: str, phase: str) -> None:
    """Back up the output files for `group_id` that the chosen `phase` will regenerate.

    - analyze:  back up analysis + critic (Phase 1 + 1.5 re-runs).
    - refactor: back up everything except analysis + critic (Phase 2 through Phase 3).
    - all:      everything above.
    """
    analysis_path = os.path.join(output_dir, f"{group_id}_analysis.json")
    critic_path = os.path.join(output_dir, f"{group_id}_critic.json")
    initial_path = os.path.join(output_dir, f"{group_id}_initial.json")
    final_path = os.path.join(output_dir, f"{group_id}_final.json")
    iter_paths = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(f"{group_id}_refactor_iter_") and f.endswith(".json")
    )
    correctness_critic_paths = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(f"{group_id}_correctness_critic_") and f.endswith(".json")
    )
    diagnose_paths = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith(f"{group_id}_diagnose_") and f.endswith(".json")
    )
    if phase == "analyze":
        targets = [analysis_path, critic_path]
    elif phase == "refactor":
        targets = [
            initial_path, final_path,
            *iter_paths, *correctness_critic_paths, *diagnose_paths,
        ]
    elif phase == "all":
        targets = [
            analysis_path, critic_path, initial_path, final_path,
            *iter_paths, *correctness_critic_paths, *diagnose_paths,
        ]
    else:
        return

    for p in targets:
        bak = _backup_file(p)
        if bak:
            print(f"  Backed up {os.path.basename(p)} -> {os.path.basename(bak)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LLM refactoring pipeline: Analyze + Critic → Refactor → Correctness Audit → Propagation."
    )
    parser.add_argument("--subgraph-dir", required=True, help="Directory containing extracted subgraph JSONs")
    parser.add_argument("--output-dir", default=None, help="Output directory for logs")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model for all phases (default: gpt-5.4)")
    parser.add_argument(
        "--phase", default="all", choices=["analyze", "refactor", "all"],
        help=(
            "'analyze' = Phase 1 analysis + Phase 1.5 performance critic; "
            "'refactor' = Phase 2 + correctness audit + Phase 3 propagation (requires cached analysis); "
            "'all' = analyze → refactor (default: all)"
        ),
    )
    parser.add_argument("--ignore-immutable", action="store_true", help="Allow LLM to change immutable models")
    parser.add_argument("--max-iterations", type=int, default=1, help="Max refactoring iterations (Phase 2 only, default: 1)")
    parser.add_argument("--groups", nargs="*", default=None, help="Specific group IDs to process (default: all)")
    parser.add_argument("--parallel-workers", type=int, default=4, help="Max parallel LLM calls for propagation and cross-subgraph processing (default: 4)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip groups that already have a _final.json in the output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only the first group with 1 iteration")
    parser.add_argument(
        "--manifest",
        default="useful_files/tuva_dbt_run_official_history/original_0415_0333/manifest.json",
        help="Path to a dbt manifest.json (pre-refactor) used to render {{ ref() }} in diagnostic SQL.",
    )
    parser.add_argument(
        "--bytes-cap",
        type=int,
        default=10 * (1024 ** 3),
        help="Max bytes a single correctness diagnostic may scan (default: 10 GiB). Over this → skip.",
    )
    parser.add_argument(
        "--strict-cost",
        action="store_true",
        help="If set, correctness diagnostics skipped for bytes-cap are treated as FAILED (default: pass with warning).",
    )
    parser.add_argument(
        "--max-diagnostic-attempts",
        type=int,
        default=3,
        help=(
            "Per-strategy attempt budget (initial try + retries after a failed gate). "
            "Retries do NOT consume a --max-iterations slot. Default: 3."
        ),
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = f"logs/refactored_slot_aware/{timestamp}/"
    os.makedirs(output_dir, exist_ok=True)

    lm = OpenAI(model=args.model)

    # discover available subgraph files
    available = []
    for fname in sorted(os.listdir(args.subgraph_dir)):
        if fname.endswith(".json"):
            available.append(fname.replace(".json", ""))

    selected = args.groups if args.groups else available

    if args.debug:
        if not args.groups:
            selected = selected[:1]
        args.max_iterations = 1
        print(f"DEBUG MODE: processing {selected}, max_iterations=1")

    # --- Process subgraphs in parallel ---
    def _process_subgraph(group_id):
        """Process a single subgraph through all phases. Returns group_id for logging."""
        subgraph_path = os.path.join(args.subgraph_dir, f"{group_id}.json")
        if not os.path.exists(subgraph_path):
            print(f"WARNING: {subgraph_path} not found, skipping")
            return group_id

        if args.skip_existing:
            final_path = os.path.join(output_dir, f"{group_id}_final.json")
            if os.path.exists(final_path):
                print(f"SKIP: {group_id} — _final.json already exists (--skip-existing)")
                return group_id

        analysis_path = os.path.join(output_dir, f"{group_id}_analysis.json")
        if args.phase == "refactor" and not os.path.exists(analysis_path):
            print(f"SKIP: {group_id} — --phase refactor requires cached analysis at {analysis_path}")
            return group_id

        print(f"\n{'='*60}")
        print(f"Processing: {group_id}  (--phase {args.phase})")
        print(f"{'='*60}")

        _backup_group_files(output_dir, group_id, args.phase)

        try:
            with open(subgraph_path) as f:
                subgraph_data = json.load(f)

            dag_text = build_dag_text(subgraph_data, ignore_immutable=args.ignore_immutable)
            all_sql = collect_model_sql(subgraph_data)
            immutable_uids = list(subgraph_data["subgraph"].get("has_external_children", {}).keys())

            # ================================================================
            # Phase 1: Analyze
            # ================================================================
            analysis_cache_path = os.path.join(output_dir, f"{group_id}_analysis.json")

            if os.path.exists(analysis_cache_path):
                print(f"  Phase 1: Loading cached analysis")
                with open(analysis_cache_path) as f:
                    analysis_data = json.load(f)
                analysis = analysis_data["analysis"]
            else:
                print(f"  Phase 1: Analyzing {len(all_sql)} models...")
                analyze_prompt = build_analyze_prompt(
                    dag_text, all_sql, immutable_uids, ignore_immutable=args.ignore_immutable,
                )

                start = time.time()
                resp = lm.chat_messages(analyze_prompt)
                elapsed = time.time() - start

                try:
                    analysis = parse_json_output(resp)
                except Exception as e:
                    print(f"  ERROR parsing analysis: {e}")
                    analysis = {"model_families": [], "standalone_models": list(all_sql.keys()),
                                "costly_patterns": [], "optimization_opportunities": []}

                with open(analysis_cache_path, "w") as f:
                    json.dump({
                        "analysis": analysis,
                        "model": resp.model,
                        "cost": resp.cost,
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                        "total_tokens": resp.total_tokens,
                        "elapsed_seconds": elapsed,
                    }, f, indent=2)

                print(f"    ${resp.cost:.4f} | {elapsed:.1f}s | {resp.total_tokens} tokens")
                families = analysis.get("model_families", [])
                all_opps = analysis.get("optimization_opportunities", [])
                print(f"    Found {len(families)} model families, {len(all_opps)} optimization opportunities")

            # ================================================================
            # Phase 1.5: Performance Critic
            # ================================================================
            all_opps = analysis.get("optimization_opportunities", [])
            critic_cache_path = os.path.join(output_dir, f"{group_id}_critic.json")

            if not all_opps:
                print(f"  Phase 1.5: No opportunities to review")
            elif os.path.exists(critic_cache_path):
                print(f"  Phase 1.5: Loading cached critic review")
                with open(critic_cache_path) as f:
                    critic_data = json.load(f)
                critic_result = critic_data["critic"]
            else:
                print(f"  Phase 1.5: Adversarial review of {len(all_opps)} opportunities...")
                critic_prompt = build_critic_prompt(dag_text, all_sql, analysis)

                start = time.time()
                critic_resp = lm.chat_messages(critic_prompt)
                elapsed = time.time() - start

                try:
                    critic_result = parse_json_output(critic_resp)
                except Exception as e:
                    print(f"  ERROR parsing critic: {e}")
                    critic_result = {"reviews": []}

                with open(critic_cache_path, "w") as f:
                    json.dump({
                        "critic": critic_result,
                        "model": critic_resp.model,
                        "cost": critic_resp.cost,
                        "prompt_tokens": critic_resp.prompt_tokens,
                        "completion_tokens": critic_resp.completion_tokens,
                        "total_tokens": critic_resp.total_tokens,
                        "elapsed_seconds": elapsed,
                    }, f, indent=2)

                print(f"    ${critic_resp.cost:.4f} | {elapsed:.1f}s | {critic_resp.total_tokens} tokens")

            # Filter opportunities based on critic verdicts
            if all_opps:
                reviews = critic_result.get("reviews", [])
                review_by_idx = {r.get("strategy_index", -1): r for r in reviews}

                approved_opps = []
                for i, opp in enumerate(all_opps):
                    if i not in review_by_idx:
                        print(f"    NO REVIEW (defaulted to reject): {opp['strategy'][:80]}")
                        continue
                    review = review_by_idx[i]
                    verdict = review.get("verdict", "reject")
                    if verdict == "approve":
                        approved_opps.append(opp)
                        print(f"    APPROVED: {opp['strategy'][:80]}")
                    elif verdict == "conditional":
                        opp["critic_condition"] = review.get("condition", "")
                        approved_opps.append(opp)
                        print(f"    CONDITIONAL: {opp['strategy'][:80]}")
                        print(f"      Condition: {review.get('condition', '')[:120]}")
                    else:
                        print(f"    REJECTED: {opp['strategy'][:80]}")
                        print(f"      Reason: {review.get('strongest_counterargument', '')[:120]}")

                analysis["optimization_opportunities"] = approved_opps
                print(f"    {len(approved_opps)}/{len(all_opps)} opportunities passed critic review")

            if args.phase == "analyze":
                if os.path.exists(critic_cache_path):
                    print(f"  Phase 1 + Phase 1.5 complete. Review at: {analysis_cache_path} and {critic_cache_path}")
                else:
                    print(f"  Phase 1 complete (no opportunities -> no critic review). Review at: {analysis_cache_path}")
                return group_id

            # ================================================================
            # Shared: Partition SQL based on analysis
            # ================================================================
            representative_uids = set()
            family_map = {}  # family_name -> family_info
            for fam in analysis.get("model_families", []):
                rep_uid = fam.get("representative")
                if not rep_uid:
                    print(f"  WARNING: family '{fam.get('family_name', '?')}' has no representative; skipping.")
                    continue
                representative_uids.add(rep_uid)
                family_map[fam.get("family_name", "")] = fam

            standalone_uids = set(analysis.get("standalone_models", []))
            immutable_set = set(immutable_uids) if not args.ignore_immutable else set()

            representative_sql = {uid: all_sql[uid] for uid in representative_uids if uid in all_sql}
            standalone_sql_map = {uid: all_sql[uid] for uid in standalone_uids if uid in all_sql and uid not in immutable_set}
            immutable_sql_map = {uid: all_sql[uid] for uid in immutable_set if uid in all_sql}

            # ================================================================
            # Phase 2: Refactor (no self-diagnostics, no propagation)
            # Produces _initial.json
            # ================================================================

            def _run_refactor_phase(previous_plans_arg, iteration_label="latest"):
                """Single Phase 2 LLM call. Returns (parsed, project_plan, resp) or (None, None, None) on error."""
                refactor_prompt = build_refactor_prompt(
                    dag_text, analysis,
                    representative_sql, standalone_sql_map, immutable_sql_map,
                    ignore_immutable=args.ignore_immutable,
                    previous_plans=previous_plans_arg,
                )

                start = time.time()
                resp = lm.chat_messages(refactor_prompt)
                elapsed = time.time() - start

                try:
                    parsed_try = parse_json_output(resp)
                    project_plan_try = parsed_try.get("project_plan", {})
                except Exception as e:
                    print(f"    ERROR parsing refactoring response: {e}")
                    return None, None, None

                perf = project_plan_try.get("performance_analysis", {})
                print(f"    ${resp.cost:.4f} | {elapsed:.1f}s | {resp.total_tokens} tokens")
                if perf:
                    print(f"    Performance: {perf.get('estimated_reduction', 'N/A')}")

                # Save refactor iter log
                log_data = {
                    "phase": "refactor",
                    "group_id": group_id,
                    "llm_model": resp.model,
                    "cost": resp.cost,
                    "elapsed_time_seconds": elapsed,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "total_tokens": resp.total_tokens,
                    "reasoning_tokens": resp.reasoning_tokens,
                    "cached_input_tokens": resp.cached_input_tokens,
                    "analysis": analysis,
                    "prompt_messages": refactor_prompt,
                    "response_content": parsed_try,
                }
                log_path = os.path.join(output_dir, f"{group_id}_refactor_iter_{iteration_label}.json")
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=2)

                return parsed_try, project_plan_try, resp

            initial_json_path = os.path.join(output_dir, f"{group_id}_initial.json")

            if args.phase in ("refactor", "all"):
                print(f"  Phase 2: Generating refactoring proposal...")
                parsed_refactor, project_plan_refactor, _ = _run_refactor_phase([], iteration_label="initial")
                if parsed_refactor is None:
                    return group_id

                initial_output = {
                    "project_plan": project_plan_refactor,
                    "models": parsed_refactor.get("models", []),
                }
                with open(initial_json_path, "w") as f:
                    json.dump(initial_output, f, indent=2)
                print(f"  Phase 2 complete: {initial_json_path}")


            # ================================================================
            # Audit: Phase 2.6 correctness critic + Phase 2.7 diagnostics
            # + Phase 3 propagation. Reads _initial.json, produces _final.json
            # ================================================================

            def _run_audit_phase(initial_data, iteration_label="latest"):
                """Run correctness audit gates. Returns (passed, project_plan, models, failure_entry_or_None)."""
                audit_project_plan = initial_data.get("project_plan", {})
                refactored_models_list = initial_data.get("models", []) or []

                # --- Phase 2.6: Adversarial correctness critic ---
                print(f"  Phase 2.6: Correctness critic...")
                cc_prompt = build_correctness_critic_prompt(
                    subgraph_data, refactored_models_list, all_sql,
                    optimization_opportunities=analysis.get("optimization_opportunities", []),
                    project_plan=audit_project_plan,
                )
                cc_start = time.time()
                cc_resp = lm.chat_messages(cc_prompt)
                cc_elapsed = time.time() - cc_start
                try:
                    cc_parsed = parse_json_output(cc_resp)
                except Exception as e:
                    print(f"    ERROR parsing correctness critic output: {e}; treating as no diagnostics.")
                    cc_parsed = {"correctness_diagnostics": []}

                diagnostics = cc_parsed.get("correctness_diagnostics", []) or []

                approved_opps = analysis.get("optimization_opportunities", [])
                for d in diagnostics:
                    idx = d.get("opportunity_index")
                    if idx is not None and 0 <= idx < len(approved_opps):
                        d["opportunity_strategy"] = approved_opps[idx].get("strategy", "")

                cc_log = {
                    "diagnostics_count": len(diagnostics),
                    "diagnostics": diagnostics,
                    "llm_model": cc_resp.model,
                    "cost": cc_resp.cost,
                    "elapsed_time_seconds": cc_elapsed,
                    "prompt_tokens": cc_resp.prompt_tokens,
                    "completion_tokens": cc_resp.completion_tokens,
                    "total_tokens": cc_resp.total_tokens,
                    "reasoning_tokens": cc_resp.reasoning_tokens,
                    "cached_input_tokens": cc_resp.cached_input_tokens,
                }
                cc_log_path = os.path.join(output_dir, f"{group_id}_correctness_critic_{iteration_label}.json")
                with open(cc_log_path, "w") as f:
                    json.dump(cc_log, f, indent=2)
                print(f"    Correctness critic: ${cc_resp.cost:.4f} | {cc_elapsed:.1f}s | {len(diagnostics)} diagnostic(s)")

                # --- Phase 2.7: Execute diagnostics ---
                if diagnostics:
                    print(f"  Phase 2.7: Executing {len(diagnostics)} correctness diagnostic(s) against BigQuery...")
                all_passed, diag_results, failures = run_post_refactor_diagnostics(
                    diagnostics,
                    manifest_path=args.manifest,
                    bytes_cap=args.bytes_cap,
                    strict_cost=args.strict_cost,
                )

                diagnose_record = {
                    "all_passed": all_passed,
                    "results": diag_results,
                    "failures": failures,
                }
                diag_log_path = os.path.join(output_dir, f"{group_id}_diagnose_{iteration_label}.json")
                with open(diag_log_path, "w") as f:
                    json.dump(diagnose_record, f, indent=2)

                if not all_passed:
                    entry = {
                        "failure_kind": "correctness",
                        "summary": audit_project_plan.get("summary", ""),
                        "performance_analysis": audit_project_plan.get("performance_analysis", {}),
                        "diagnostic_failures": failures,
                    }
                    return False, audit_project_plan, refactored_models_list, entry

                print(f"    All diagnostics PASSED.")
                return True, audit_project_plan, refactored_models_list, None

            # --- Run audit retry loop (--phase refactor or all) ---

            if args.phase in ("refactor", "all"):
                # Retry loop: audit failures feed back into refactor
                initial_data = initial_output  # from the refactor phase above
                previous_plans = []
                audit_passed = False

                refactor_failed = False
                for strategy_iter in range(args.max_iterations):
                    print(f"  Audit: Strategy attempt {strategy_iter+1}/{args.max_iterations}...")

                    for attempt_idx in range(args.max_diagnostic_attempts):
                        iter_label = f"s{strategy_iter}_a{attempt_idx}"
                        audit_passed, project_plan_try, _, failure = _run_audit_phase(initial_data, iteration_label=iter_label)
                        if audit_passed:
                            project_plan = project_plan_try
                            break

                        # Audit failed — feed failure back into refactor
                        previous_plans.append(failure)
                        retry_label = f" (retry {attempt_idx+1}/{args.max_diagnostic_attempts})"
                        print(f"  Re-running Phase 2{retry_label} with audit feedback...")
                        parsed_retry, pp_retry, _ = _run_refactor_phase(previous_plans, iteration_label=iter_label)
                        if parsed_retry is None:
                            refactor_failed = True
                            break
                        initial_data = {
                            "project_plan": pp_retry,
                            "models": parsed_retry.get("models", []),
                        }
                        with open(initial_json_path, "w") as f:
                            json.dump(initial_data, f, indent=2)

                    if audit_passed:
                        break
                    if refactor_failed:
                        print(f"    Refactor parse failure; aborting retry loop.")
                        break

                    has_more = initial_data.get("project_plan", {}).get("has_more_strategies", True)
                    if not has_more:
                        print(f"    No further distinct strategies.")
                        break

                if not audit_passed:
                    print(f"  All audit attempts FAILED; reverting every model to original SQL.")
                    reverted_models = [
                        {
                            "uid": uid,
                            "rewritten": False,
                            "is_new": False,
                            "removed": False,
                            "materialized": None,
                            "rewritten_sql": sql,
                            "description": "Reverted to original — no refactor plan passed audit.",
                        }
                        for uid, sql in all_sql.items()
                    ]
                    combined_output = {
                        "project_plan": {
                            "summary": "All refactor attempts failed audit; group left unchanged.",
                            "correctness_status": "rejected_all_unchanged",
                            "previous_plans": previous_plans,
                        },
                        "models": reverted_models,
                    }
                    combined_path = os.path.join(output_dir, f"{group_id}_final.json")
                    with open(combined_path, "w") as f:
                        json.dump(combined_output, f, indent=2)
                    print(f"  Final output: {len(reverted_models)} models (all unchanged) -> {combined_path}")
                    return group_id

            # ================================================================
            # Phase 3: Propagate (runs after audit passes)
            # ================================================================
            parsed = initial_data
            propagation_info = project_plan.get("family_propagation", [])
            refactored_models = {m["uid"]: m for m in parsed.get("models", [])}
            propagated_models = []
            already_propagated = set()

            llm_tasks = []

            for prop in propagation_info:
                family_name = prop.get("family_name", "")
                rep_uid = prop.get("representative", "")
                change_desc = prop.get("change_description", "")

                fam_info = family_map.get(family_name)
                if fam_info is None:
                    print(f"  WARNING: family_propagation entry references unknown family '{family_name}'; skipping.")
                    continue
                target_members = fam_info.get("other_members", [])
                key_diffs = fam_info.get("key_differences", "")

                refactored_rep = refactored_models.get(rep_uid, {})
                refactored_rep_sql = refactored_rep.get("rewritten_sql", "")
                original_rep_sql = all_sql.get(rep_uid, "")

                if not refactored_rep_sql or not target_members:
                    continue

                remaining = [m for m in target_members if m not in already_propagated]
                if not remaining:
                    continue

                print(f"  Phase 3: Propagating {family_name} ({len(remaining)} members via LLM)")

                for member_uid in remaining:
                    member_sql = all_sql.get(member_uid, "")
                    if not member_sql:
                        continue

                    llm_tasks.append({
                        "family_name": family_name,
                        "rep_uid": rep_uid,
                        "refactored_rep_sql": refactored_rep_sql,
                        "original_rep_sql": original_rep_sql,
                        "member_uid": member_uid,
                        "member_sql": member_sql,
                        "key_diffs": key_diffs,
                        "change_desc": change_desc,
                    })
                    already_propagated.add(member_uid)

            short_name_to_uid = {uid.split('.')[-1]: uid for uid in all_sql}
            for m in parsed.get("models", []):
                uid = m.get("uid")
                if uid:
                    short_name_to_uid[uid.split('.')[-1]] = uid

            def _upstream_refs_sql_for(refactored_sql: str) -> dict:
                out = {}
                for match in re.finditer(
                    r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}",
                    refactored_sql,
                ):
                    short = match.group(1)
                    if short in out:
                        continue
                    uid = short_name_to_uid.get(short)
                    if uid is None:
                        continue
                    rm = refactored_models.get(uid, {})
                    sql = rm.get("rewritten_sql") or all_sql.get(uid) or ""
                    if sql:
                        out[short] = sql
                return out

            for task in llm_tasks:
                task["upstream_refs_sql"] = _upstream_refs_sql_for(task["refactored_rep_sql"])
                member_refs = _upstream_refs_sql_for(task["member_sql"])
                for k, v in member_refs.items():
                    if k not in task["upstream_refs_sql"]:
                        task["upstream_refs_sql"][k] = v

            def _propagate_one(task):
                prop_prompt = build_propagate_prompt(
                    task["family_name"], task["rep_uid"],
                    task["refactored_rep_sql"], task["original_rep_sql"],
                    task["member_uid"], task["member_sql"],
                    task["key_diffs"], task["change_desc"],
                    upstream_refs_sql=task.get("upstream_refs_sql"),
                )
                t0 = time.time()
                prop_resp = lm.chat_messages(prop_prompt)
                elapsed = time.time() - t0
                return task["member_uid"], task["rep_uid"], prop_resp, elapsed

            if llm_tasks:
                print(f"  Phase 3: Launching {len(llm_tasks)} LLM propagations in parallel (workers={args.parallel_workers})")
                with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
                    futures = {executor.submit(_propagate_one, t): t for t in llm_tasks}
                    for future in as_completed(futures):
                        task = futures[future]
                        member_uid = task["member_uid"]
                        try:
                            uid, rep_uid, prop_resp, elapsed = future.result()
                            prop_parsed = parse_json_output(prop_resp)
                            propagated_models.append({
                                "uid": uid,
                                "rewritten": True,
                                "is_new": False,
                                "removed": False,
                                "materialized": None,
                                "rewritten_sql": prop_parsed.get("rewritten_sql", ""),
                                "description": prop_parsed.get("description", f"Propagated from {rep_uid}"),
                            })
                            already_propagated.add(uid)
                            print(f"      {uid}: ${prop_resp.cost:.4f} | {elapsed:.1f}s")
                        except Exception as e:
                            print(f"      ERROR propagating {member_uid}: {e}")
                            propagated_models.append({
                                "uid": member_uid,
                                "rewritten": False,
                                "is_new": False,
                                "removed": False,
                                "materialized": None,
                                "rewritten_sql": task["member_sql"],
                                "description": f"Propagation failed: {e}",
                            })

            output_uids = set(refactored_models.keys()) | already_propagated
            component_set = set(subgraph_data["subgraph"].get("component_nodes", []))
            for fam in family_map.values():
                for member_uid in fam.get("other_members", []):
                    if member_uid not in output_uids and member_uid in all_sql and member_uid in component_set:
                        propagated_models.append({
                            "uid": member_uid,
                            "rewritten": False,
                            "is_new": False,
                            "removed": False,
                            "materialized": None,
                            "rewritten_sql": all_sql[member_uid],
                            "description": "Unchanged (family not refactored)",
                        })
                        output_uids.add(member_uid)

            all_output_models = parsed.get("models", []) + propagated_models

            combined_output = {
                "project_plan": project_plan,
                "models": all_output_models,
            }
            combined_path = os.path.join(output_dir, f"{group_id}_final.json")
            with open(combined_path, "w") as f:
                json.dump(combined_output, f, indent=2)

            print(f"  Final output: {len(all_output_models)} models -> {combined_path}")

        except Exception as e:
            print(f"  ERROR processing {group_id}: {e}")
            return group_id

        return group_id

    # --- Dispatch subgraphs ---
    if args.parallel_workers > 1 and len(selected) > 1:
        print(f"\nProcessing {len(selected)} subgraphs with up to {args.parallel_workers} parallel workers")
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            futures = {executor.submit(_process_subgraph, gid): gid for gid in selected}
            for future in as_completed(futures):
                gid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  UNHANDLED ERROR in {gid}: {e}")
    else:
        for group_id in selected:
            _process_subgraph(group_id)

    print(f"\nAll done. Output in {output_dir}")


if __name__ == "__main__":
    main()