import os
import json
import argparse

from apply_refactoring import (
    run_git,
    infer_new_model_path,
    update_yaml_materialization,
    add_yaml_model_entry,
)
from utility.analyze import load_manifest


DEFAULT_MANIFEST = "useful_files/tuva_dbt_run_official_history/original_0211_0752/manifest.json"
DEFAULT_PROJECT_ROOT = "../dbt_project"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply selected refactorings from a selection JSON to a single dbt project branch."
    )
    parser.add_argument(
        "selection_json",
        help="Path to the *_selection.json file from select_best_combination.py",
    )
    parser.add_argument(
        "--refactored-dir",
        required=True,
        help="Directory containing group_*_final.json files (e.g. logs/refactored/0329_0400)",
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST,
        help=f"Path to dbt manifest.json (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--project-root",
        default=DEFAULT_PROJECT_ROOT,
        help=f"Path to dbt project root (default: {DEFAULT_PROJECT_ROOT})",
    )
    parser.add_argument(
        "--base-branch",
        default="original",
        help="Base branch to create the combined branch from (default: original)",
    )
    parser.add_argument(
        "--branch-name",
        default=None,
        help="Name for the new branch (default: auto-generated from refactored-dir)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # load selection
    with open(args.selection_json) as f:
        selection = json.load(f)

    selected = selection["selected"]
    if not selected:
        print("No refactorings selected, nothing to do.")
        return

    print(f"Selection: {len(selected)} groups to apply")
    for s in selected:
        label = s.get('candidate', s.get('branch', ''))
        print(f"  {s['group']:12s}  delta = {s['delta']:>14,.2f}  ({label})")

    # load manifest
    manifest = load_manifest(args.manifest_path)
    manifest_nodes = manifest["nodes"]

    # map groups to final.json files
    group_files = {}
    for entry in selected:
        group_id = entry["group"]
        path = os.path.join(args.refactored_dir, f"{group_id}_final.json")
        if not os.path.exists(path):
            print(f"ERROR: Missing {path} for selected group {group_id}")
            return
        group_files[group_id] = path

    # pre-flight: uid collisions across selected groups must be impossible
    # (enforced upstream by the ILP in select_best_combination.py). If one
    # reaches here, abort before touching git/FS rather than silently overwrite.
    uid_to_group = {}
    collisions = {}
    for group_id, filepath in group_files.items():
        with open(filepath) as f:
            data = json.load(f)
        for m in data.get("models", []):
            if not (m.get("rewritten") or m.get("is_new") or m.get("removed")):
                continue
            uid = m["uid"]
            if uid in uid_to_group and uid_to_group[uid] != group_id:
                collisions.setdefault(uid, [uid_to_group[uid]]).append(group_id)
            else:
                uid_to_group[uid] = group_id

    if collisions:
        print("ERROR: uid collisions across selected groups:")
        for uid, gids in sorted(collisions.items()):
            print(f"  {uid}: {gids}")
        print("Run compute_conflict_graph.py and re-run select_best_combination.py with --conflicts.")
        return

    # branch setup
    project_root = args.project_root
    dir_name = os.path.basename(os.path.normpath(args.refactored_dir))
    branch_name = args.branch_name or f"selected_{dir_name}_{len(selected)}_groups"

    print(f"\nCreating branch '{branch_name}' from '{args.base_branch}'")
    if not run_git(["git", "checkout", args.base_branch], cwd=project_root):
        print("ERROR: Failed to checkout base branch")
        return
    if not run_git(["git", "checkout", "-B", branch_name], cwd=project_root):
        print(f"ERROR: Failed to create branch {branch_name}")
        return

    # apply all groups
    total_new = 0
    total_mod = 0
    total_rm = 0

    for group_id, filepath in group_files.items():
        print(f"\n--- Applying {group_id} ---")

        with open(filepath) as f:
            data = json.load(f)

        project_plan = data.get("project_plan", {})
        models = data.get("models", [])

        for m in models:
            uid = m.get("uid", "")
            is_new = m.get("is_new", False)
            is_rewritten = m.get("rewritten", False)
            is_removed = m.get("removed", False)
            materialized = m.get("materialized")
            sql = m.get("rewritten_sql", "")
            short_name = uid.split(".")[-1]

            if not (is_rewritten or is_new or is_removed):
                continue

            # determine file path
            if uid in manifest_nodes:
                relative_path = manifest_nodes[uid]["original_file_path"]
            elif is_new:
                relative_path = infer_new_model_path(uid, project_plan, manifest_nodes)
                print(f"  [PATH] Inferred: {relative_path}")
            else:
                if not is_removed:
                    print(f"  [WARN] Unknown model {uid}, skipping")
                continue

            full_path = os.path.join(project_root, relative_path)

            # apply changes
            if is_removed:
                if os.path.exists(full_path):
                    os.remove(full_path)
                    print(f"  [RM]  {relative_path}")
                    total_rm += 1

            elif is_rewritten or is_new:
                if not sql:
                    print(f"  [WARN] No SQL for {uid}, skipping")
                    continue
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(sql)

                if is_new:
                    print(f"  [NEW] {relative_path}")
                    total_new += 1
                else:
                    print(f"  [MOD] {relative_path}")
                    total_mod += 1

                # YAML materialization
                if materialized:
                    if is_new:
                        add_yaml_model_entry(relative_path, short_name, materialized, project_root)
                    else:
                        if update_yaml_materialization(project_root, short_name, materialized):
                            print(f"  [MAT] {short_name} -> {materialized}")
                        else:
                            add_yaml_model_entry(relative_path, short_name, materialized, project_root)

    # commit
    total_changes = total_new + total_mod + total_rm
    if total_changes > 0:
        group_ids = sorted(group_files.keys())
        commit_msg = (
            f"Apply selected refactorings: {', '.join(group_ids)}\n\n"
            f"Groups: {', '.join(group_ids)}\n"
            f"New: {total_new}, Modified: {total_mod}, Removed: {total_rm}\n"
            f"Source: {os.path.basename(args.selection_json)}"
        )
        run_git(["git", "add", "-A"], cwd=project_root)
        run_git(["git", "commit", "-m", commit_msg], cwd=project_root)

    # summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Branch:   {branch_name}")
    print(f"Groups:   {len(group_files)}")
    print(f"New:      {total_new}")
    print(f"Modified: {total_mod}")
    print(f"Removed:  {total_rm}")
    print(f"Total:    {total_changes}")

    if total_changes == 0:
        print("\nNo changes to commit.")
    else:
        print(f"\nCommitted to branch '{branch_name}'")


if __name__ == "__main__":
    main()
