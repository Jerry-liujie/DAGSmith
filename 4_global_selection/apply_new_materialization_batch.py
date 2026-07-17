import os
import json
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply new materializations to dbt project branches in batch."
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Prefix to match subfolders in run-history-folder "
             "(e.g. 'refactor_0329_0400_group_')",
    )
    parser.add_argument(
        "--proj-root",
        default="../dbt_project",
        help="Path to the dbt project root",
    )
    parser.add_argument(
        "--new-materialization-folder",
        default="useful_files/tuva_analysis/for_materialization_tuning/new_materializations",
        help="Folder containing new materialization JSON files",
    )
    parser.add_argument(
        "--run-history-folder",
        default="useful_files/tuva_dbt_run_official_history",
        help="Folder containing dbt run history subfolders",
    )
    parser.add_argument(
        "--branch-suffix",
        default="_ilp",
        help="Suffix for the new git branch name",
    )
    return parser.parse_args()


args = parse_args()
run_history_folder = args.run_history_folder
new_materialization_folder = args.new_materialization_folder
proj_root = args.proj_root

def run_command(cmd, cwd=None):
    """Helper to run shell commands in a specific directory."""
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

for filename in sorted(os.listdir(new_materialization_folder)):
    if not filename.startswith(args.prefix) or not filename.endswith("_new_materialization.json"):
        continue

    subfolder = filename.removesuffix("_new_materialization.json")
    base_branch_name = subfolder.split("_compile_")[0]

    new_branch_name = f"{base_branch_name}{args.branch_suffix}"
    
    # Full path to the JSON file inside the subfolder
    json_path = os.path.join(new_materialization_folder, filename)

    try:
        # 1. Checkout to the base branch (ensure it exists)
        run_command(f"git checkout {base_branch_name}", cwd=proj_root)
        
        # 2. Create and switch to the new _ilp branch (error if it already exists)
        result = subprocess.run(
            f"git branch --list {new_branch_name}", shell=True, cwd=proj_root,
            capture_output=True, text=True
        )
        if result.stdout.strip():
            print(f"Error: branch '{new_branch_name}' already exists. Exiting.")
            exit(1)
        run_command(f"git checkout -b {new_branch_name}", cwd=proj_root)
        
        # 3. Run the materialization script
        # Note: We pass the full path to the JSON file as the first argument
        run_command(f"python apply_new_materialization.py {json_path} {proj_root}")
        
        # 4. Stage and commit 
        ## important!!!!! I am not sure if this works correctly
        run_command("git add -u", cwd=proj_root)
        run_command(f'git commit -m "{subfolder} apply new materialization"', cwd=proj_root)
        
        print(f"Successfully processed {subfolder} on branch {new_branch_name}")

    except subprocess.CalledProcessError as e:
        print(f"Error processing {subfolder}: {e}")
        continue
    
    
    


                    
