# DAGSmith — Core Pipeline

DAGSmith is a dependency-aware rewriting system for dbt-style SQL pipelines. Given a
SQL pipeline expressed as a DAG of models, it produces an optimized DAG that runs
cheaper (less warehouse slot time) while preserving results. It combines three coupled
optimization dimensions:

1. **Graph refactoring** — restructure the DAG to remove or share expensive work
   (non-local reuse, redundant/mistimed join pruning, work placement).
2. **Materialization tuning** — decide, per model, whether it should be a stored
   *table* or an inlined *view*, scoring each refactoring under its best assignment.
3. **Refresh-frequency tuning** — align each model's execution rate with how often its
   inputs change and its outputs are consumed.

This folder is a curated, reference-only view containing just the **core files** that
implement the end-to-end pipeline, organized by stage. Evaluation scripts, baselines,
backups, and dataset-specific duplicates from the full research repo are omitted.

---

## Pipeline at a glance

```
SQL pipeline DAG P
      │
      ▼
 [1] Candidate Identification   → ranked model groups → subgraphs
      │
      ▼
 [2] LLM Refactoring            → verified refactoring candidates
      │
      ▼
 [3] Materialization Tuning     → per-candidate best table/view + cost delta Δc
      │
      ▼
 [4] Global Selection           → compatible subset + final retune → optimized DAG P*
      │
      ▼
 [5] Frequency-Aware Optimization (feeds weights / reschedule / split hints into 1,3,4)
```

The four main stages correspond to the four stages of the approach; the frequency
dimension feeds three of them.

---

## Directory layout

```
core_files/
├── README.md
├── utility/                     # shared library used by every stage
├── 1_candidate_identification/
│   └── dbt_column_lineage_extractor/   # dataflow / column-lineage engine
├── 2_llm_refactoring/
├── 3_materialization_tuning/
├── 4_global_selection/
└── 5_frequency/
```

---

## Stage 1 — Candidate Identification  (`1_candidate_identification/`)

Cheap structural (dataflow) analyses scan the whole DAG, rank model groups by a
fingerprint-based benefit score (reuse of shared work + pruning of unused/early work),
and extract bounded subgraphs (candidate models + their one-hop upstream parents) for
the expensive LLM stage.

The dataflow analysis itself lives in the `dbt_column_lineage_extractor/` package.
It compiles each model's SQL into a typed dataflow expression tree, resolves every
column and `ref()` back through the models that produced it (so a model boundary no
longer hides shared work), and computes structural *signatures/fingerprints* used to
group models by the operation they compute. On top of that representation it runs the
two analyses from the paper — **reuse** (find repeated expensive work to share) and
**prune** (find joins whose result is unused or computed too early) — and groups models
into ranked candidates. `extract_subgraphs.py` then turns the ranked groups into
extraction-ready subgraphs.

| File | Role |
|------|------|
| `dbt_column_lineage_extractor/` | The dataflow / column-lineage engine that powers candidate identification (see per-module table below). |
| `extract_subgraphs.py` | Extract a refactoring subgraph for each ranked model group (component nodes + one-hop parents), skipping groups over a node-count cap. Reads a ranked-groups JSON + a dbt `manifest.json`. |
| `extract_subgraph_all.py` | Extract a single subgraph over the entire project (used for whole-DAG passes). |

Inside `dbt_column_lineage_extractor/`:

| Module | Role |
|--------|------|
| `extractor.py` | Core engine: builds per-model scopes from compiled SQL, resolves cross-model column lineage, groups models by dataflow similarity (`group_models_by_scope_similarity`), and runs the reuse (`analysis_for_reuse`) and prune (`analysis_for_prune`) analyses that emit ranked candidates. |
| `dfexpr.py` | The typed dataflow expression node types (column refs, literals, transforms, joins, predicates). |
| `dfexpr_utils.py` | Canonicalization, structural equality, and fingerprinting/hashing of dataflow expressions (the "signatures" used to detect shared work). |
| `dbt_info.py` | Parse dbt artifacts (`manifest.json` / optional catalog) into model nodes with compiled SQL and schema. |
| `utils.py` | SQL parsing (sqlglot) and shared helpers. |

## Stage 2 — LLM Refactoring  (`2_llm_refactoring/`)

An agentic loop of four LLM roles turns each subgraph into a concrete, verified
rewrite: a **proposer** lists opportunities, an adversarial **critic** rejects unsafe
or slower proposals, a **rewriter** emits refactored SQL, and a **verifier** runs
data-backed equivalence checks against the real warehouse before a rewrite is trusted.

| File | Role |
|------|------|
| `refactor_slot_aware.py` | The full proposer → critic → rewriter → verifier agentic refactoring loop, with the slot-time cost model embedded in the prompts. Emits per-group final refactorings. |
| `apply_refactoring.py` | Apply each accepted per-group refactoring to its own branch in the dbt project so it can be compiled and benchmarked. |

## Stage 3 — Materialization Tuning  (`3_materialization_tuning/`)

Each refactoring's benefit depends on which of its nodes are materialized. A learned
cost model predicts per-node cardinality and slot time; **iterative local
linearization** then searches the table-or-view assignment space via a sequence of
small ILPs to find each candidate's cheapest materialization.

| File | Role |
|------|------|
| `extract_features_for_training.py` | Parse structural SQL features from the original pipeline's compiled models and pair them with measured runtimes to build the cost-model training set. |
| `train_huber_regression.py` | Train the two robust (Huber) regressors — output cardinality and slot time — that form the learned cost model. |
| `extract_features_for_refactored.py` | Extract the same features for a refactored candidate's models (including brand-new models), conditioning on parents' predicted cardinalities. |
| `iterative_local_linearization.py` | Successive-approximation table/view search: linearize cost at the current assignment, solve a small ILP, re-linearize until stable. Produces each candidate's best materialization + cost. |
| `materialization_info.py` | Helper to read materialization config (table/view) out of a dbt manifest. |

## Stage 4 — Global Selection  (`4_global_selection/`)

Candidates overlap and conflict, so DAGSmith picks the compatible subset with the
largest total cost delta via an ILP, then retunes the merged DAG one final time.

| File | Role |
|------|------|
| `compute_conflict_graph.py` | Build the conflict graph between candidates (two refactorings touching the same model cannot both be applied). |
| `compute_refactoring_deltas.py` | Compute each candidate's delta: pipeline cost saved when that refactoring is applied, each side at its own best materialization. |
| `select_best_combination.py` | ILP that selects the max-total-delta, conflict-free set with at most one variant per region. |
| `apply_selected_refactorings.py` | Apply the selected combination to a single combined branch. |
| `consensus_materialization.py` | Reconcile the individual candidates' materialization choices into one consensus assignment for the combined DAG (final retune). |
| `apply_new_materialization.py` | Apply a materialization (table/view) assignment to a dbt branch. |
| `apply_new_materialization_batch.py` | Batch version: apply materialization assignments across branches. |

## Stage 5 — Frequency-Aware Optimization  (`5_frequency/`)

Derives an effective-frequency profile from source update rates + downstream output
demand, propagates it through the DAG, and applies it at three layers: (1) cost weights
for tuning/selection, (2) schedule reconciliation to drop redundant refreshes, and
(3) frequency-driven splitting hints fed back to candidate identification.

| File | Role |
|------|------|
| `frequency.py` | Core frequency propagation over the DAG (combine source-change and output-demand rates through parent/child maps). |
| `generate_frequency_profiles.py` | Generate per-node refresh-frequency profiles / scenarios for cost weighting. |
| `iterative_local_linearization_freq.py` | Frequency-weighted variant of the materialization search (Layer 1 cost weights). |
| `compute_refactoring_deltas_freq.py` | Frequency-weighted candidate deltas. |
| `consensus_materialization_freq.py` | Frequency-weighted consensus materialization / final retune. |

---

## Shared library — `utility/`

| Module | Contents |
|--------|----------|
| `analyze.py` | SQL feature extraction, manifest/DAG parent-child map building, the Huber cost-model classes/fitting, cardinality prediction, and the slot-time ILP solver. Shared by nearly every stage. |
| `subgraph.py` | `SubgraphAnalyzer` and `RefactorSubgraph` — subgraph extraction and visualization for candidate identification. |
| `gpt.py` | Thin LLM client wrapper used by the refactoring agents (reads `OPENAI_API_KEY` from the environment). |
| `__init__.py` | Package marker. |

---

## Notes

- These files are provided for reference and to document the pipeline. They expect a
  companion dbt project, a compiled `manifest.json`, and run-history artifacts that are
  not bundled here; data/output paths in the scripts are relative placeholders and will
  need to be pointed at a local environment.
- The dataflow analysis that produces the ranked model-groups input to Stage 1 is
  included as the `dbt_column_lineage_extractor/` package; it consumes a compiled dbt
  `manifest.json`.
- Because files are grouped into stage subfolders, run them with the repository root on
  `PYTHONPATH` (e.g. `PYTHONPATH=. python 1_candidate_identification/extract_subgraphs.py ...`)
  so the `utility` package resolves.
