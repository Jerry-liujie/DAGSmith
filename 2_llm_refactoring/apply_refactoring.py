import os
import json
import argparse
import subprocess
from pathlib import Path
import yaml

from utility.analyze import load_manifest


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def run_git(args, cwd):
    try:
        subprocess.run(args, check=True, text=True, capture_output=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  git error: {' '.join(args)}\n    {e.stderr.strip()}")
        return False


# ---------------------------------------------------------------------------
# Path inference for new models
# ---------------------------------------------------------------------------

def infer_new_model_path(uid, project_plan, manifest_nodes):
    """
    Determine where to write a new model's .sql file.

    Strategy:
    1. Look at reused_by from project_plan.new_shared_models
    2. Find common parent directory of those existing models
    3. Place the new model there

    Fallback: find existing models sharing the longest UID prefix.
    """
    short_name = uid.split(".")[-1]

    # Strategy 1: use reused_by from project_plan
    reused_by = []
    for nsm in project_plan.get("new_shared_models", []):
        if nsm.get("uid") == uid:
            reused_by = nsm.get("reused_by", [])
            break

    consumer_dirs = set()
    for consumer_uid in reused_by:
        node = manifest_nodes.get(consumer_uid)
        if node and "original_file_path" in node:
            consumer_dirs.add(str(Path(node["original_file_path"]).parent))

    if consumer_dirs:
        # Find common parent directory
        common = os.path.commonpath(list(consumer_dirs))
        return os.path.join(common, f"{short_name}.sql")

    # Strategy 2: find existing model with longest shared prefix
    best_match = None
    best_prefix_len = 0
    for existing_uid, node in manifest_nodes.items():
        if "original_file_path" not in node:
            continue
        # Compare the short names (after last dot)
        existing_short = existing_uid.split(".")[-1]
        # Find common prefix length
        prefix_len = len(os.path.commonprefix([short_name, existing_short]))
        if prefix_len > best_prefix_len:
            best_prefix_len = prefix_len
            best_match = node["original_file_path"]

    if best_match:
        return os.path.join(str(Path(best_match).parent), f"{short_name}.sql")

    # Last resort: put in models/ root
    return f"models/{short_name}.sql"


# ---------------------------------------------------------------------------
# YAML materialization helpers
# ---------------------------------------------------------------------------

def update_yaml_materialization(project_root, model_name, materialization):
    """Update an existing model's materialization in YAML config files."""
    project_root = Path(project_root)
    yml_files = list(project_root.glob("models/**/*.yml")) + \
                list(project_root.glob("models/**/*.yaml"))

    for yml_path in yml_files:
        with open(yml_path) as f:
            data = yaml.safe_load(f)
        if not data or "models" not in data:
            continue

        changed = False
        for model in data["models"]:
            if model.get("name") == model_name:
                if "config" not in model:
                    model["config"] = {}
                if model["config"].get("materialized") != materialization:
                    model["config"]["materialized"] = materialization
                    changed = True

        if changed:
            with open(yml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            return True
    return False


def add_yaml_model_entry(sql_file_path, model_name, materialization, project_root):
    """
    Add a new model entry to the nearest .yml file.
    Searches the SQL file's directory and parents for a .yml file.
    """
    sql_dir = Path(project_root) / Path(sql_file_path).parent
    search_dir = sql_dir

    # Walk up to find the nearest .yml
    yml_path = None
    while search_dir != Path(project_root) and search_dir != search_dir.parent:
        candidates = list(search_dir.glob("*.yml")) + list(search_dir.glob("*.yaml"))
        # Prefer files with "_models" in the name
        models_files = [c for c in candidates if "_models" in c.name]
        if models_files:
            yml_path = models_files[0]
            break
        if candidates:
            yml_path = candidates[0]
            break
        search_dir = search_dir.parent

    if yml_path is None:
        # Create a new yml file next to the SQL file
        yml_path = sql_dir / "_models.yml"
        with open(yml_path, "w") as f:
            yaml.dump({"version": 2, "models": []}, f, sort_keys=False)

    with open(yml_path) as f:
        data = yaml.safe_load(f) or {}

    if "models" not in data:
        data["models"] = []

    # Check if already exists
    for m in data["models"]:
        if m.get("name") == model_name:
            if "config" not in m:
                m["config"] = {}
            m["config"]["materialized"] = materialization
            with open(yml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
            return

    # Append new entry
    entry = {"name": model_name}
    if materialization:
        entry["config"] = {"materialized": materialization}
    data["models"].append(entry)

    with open(yml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"   [YML] Added {model_name} to {yml_path.relative_to(project_root)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply LLM refactorings from *_final.json to dbt project branches."
    )
    parser.add_argument("--refactored-dir", required=True,
                        help="Directory containing *_final.json files")
    parser.add_argument("--manifest-path", required=True,
                        help="Path to dbt manifest.json")
    parser.add_argument("--project-root", required=True,
                        help="Path to dbt project root")
    parser.add_argument("--base-branch", default="original",
                        help="Base branch to create refactoring branches from (default: original)")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Specific group IDs to apply (default: all *_final.json files)")
    args = parser.parse_args()

    project_root = args.project_root
    manifest = load_manifest(args.manifest_path)
    manifest_nodes = manifest["nodes"]

    # Extract timestamp from refactored-dir name (e.g., "0327_1809")
    dir_name = os.path.basename(os.path.normpath(args.refactored_dir))
    timestamp = dir_name  # use the whole dir name as timestamp prefix

    # Discover final files
    final_files = {}
    for fname in sorted(os.listdir(args.refactored_dir)):
        if fname.endswith("_final.json"):
            group_id = fname.replace("_final.json", "")
            final_files[group_id] = os.path.join(args.refactored_dir, fname)

    if args.groups:
        selected = [g for g in args.groups if g in final_files]
    else:
        selected = sorted(final_files.keys())

    print(f"Found {len(final_files)} final files, processing {len(selected)} groups")

    for group_id in selected:
        filepath = final_files[group_id]

        print(f"\n{'='*60}")
        print(f"Applying: {group_id}")
        print(f"{'='*60}")

        with open(filepath) as f:
            data = json.load(f)

        project_plan = data.get("project_plan", {})
        models = data.get("models", [])

        if not models:
            print("  No models to apply, skipping")
            continue

        # Branch setup
        branch_name = f"refactor_{timestamp}_{group_id}"
        print(f"  Branch: {branch_name}")

        if not run_git(["git", "checkout", args.base_branch], cwd=project_root):
            print("  Failed to checkout base branch, skipping")
            continue
        if not run_git(["git", "checkout", "-B", branch_name], cwd=project_root):
            print(f"  Failed to create branch {branch_name}, skipping")
            continue

        changes = 0

        for m in models:
            uid = m.get("uid", "")
            is_new = m.get("is_new", False)
            is_rewritten = m.get("rewritten", False)
            is_removed = m.get("removed", False)
            materialized = m.get("materialized")
            sql = m.get("rewritten_sql", "")
            short_name = uid.split(".")[-1]

            # Determine file path
            if uid in manifest_nodes:
                relative_path = manifest_nodes[uid]["original_file_path"]
            elif is_new:
                relative_path = infer_new_model_path(uid, project_plan, manifest_nodes)
                print(f"   [PATH] Inferred: {relative_path}")
            else:
                if not is_removed:
                    print(f"   [WARN] Unknown model {uid}, skipping")
                continue

            full_path = os.path.join(project_root, relative_path)

            # Apply changes
            if is_removed:
                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"   [RM] {relative_path}")
                    changes += 1

            elif is_rewritten or is_new:
                if not sql:
                    print(f"   [WARN] No SQL for {uid}, skipping")
                    continue
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(sql)
                status = "NEW" if is_new else "MOD"
                print(f"   [{status}] {relative_path}")
                changes += 1

                # Handle materialization via YAML
                if materialized:
                    if is_new:
                        add_yaml_model_entry(relative_path, short_name, materialized, project_root)
                    else:
                        if update_yaml_materialization(project_root, short_name, materialized):
                            print(f"   [MAT] {short_name} -> {materialized}")
                        else:
                            # Model not found in any YAML — add it
                            add_yaml_model_entry(relative_path, short_name, materialized, project_root)

        # Commit
        if changes > 0:
            print(f"  Committing {changes} changes...")
            run_git(["git", "add", "-A"], cwd=project_root)
            run_git(["git", "commit", "-m",
                     f"Refactor applied for {group_id}\n\nSource: {os.path.basename(filepath)}"],
                    cwd=project_root)
            print(f"  Done: branch {branch_name}")
        else:
            print("  No changes to commit")

    print("\nAll groups processed.")


if __name__ == "__main__":
    main()
