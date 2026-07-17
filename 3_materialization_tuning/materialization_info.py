#!/usr/bin/env python3
import os
import csv
from utility.analyze import extract_materializations


def main():
    manifest_path = os.path.join('useful_files/tuva_target_models/manifest.json')

    rows = extract_materializations(manifest_path)
    
    # write to CSV
    output_file = os.path.join('useful_files', 'tuva', '0520_tuva_materializations.csv')
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model_unique_id', 'materialized']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for uid, mat in rows:
            writer.writerow({'model_unique_id': uid, 'materialized': mat})


if __name__ == "__main__":
    main()
