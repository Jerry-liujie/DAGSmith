#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import yaml

# try:
#     from ruamel.yaml import YAML  # preserves comments/order
#     yaml = YAML()
#     yaml.preserve_quotes = True
# except ImportError:
#     import yaml  # PyYAML fallback (drops some formatting)


# 1. Setup paths and User Input

new_materialization_path = sys.argv[1]
proj_root_path = sys.argv[2]

PROJECT_ROOT = Path(proj_root_path)

# 2. Load the mapping (converts schema.model_name -> model_name)
with open(new_materialization_path) as f:
    raw_mapping = json.load(f)
    # Just get the last part of the name (the actual model name)
    new_m = {k.split(".")[-1]: v for k, v in raw_mapping.items()}

print(f"Loaded {len(new_m)} model updates.")

def update_file(file_path):
    """Checks a YAML file and updates materialization if needed."""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data or "models" not in data:
        return False

    changed = False
    for model in data["models"]:
        name = model.get("name")
        if name in new_m:
            # Ensure 'config' dictionary exists
            if "config" not in model:
                model["config"] = {}
            
            # Update if the value is different
            if model["config"].get("materialized") != new_m[name]:
                model["config"]["materialized"] = new_m[name]
                changed = True

    if changed:
        # Save changes back to the file
        with open(file_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        return True
    return False

# 3. Main Loop
files = list(PROJECT_ROOT.glob("models/**/*.yml")) + list(PROJECT_ROOT.glob("models/**/*.yaml"))
touched = 0

for f in files:
    if update_file(f):
        print(f"Updated: {f.name}")
        touched += 1

print(f"Done. Updated {touched} file(s).")

