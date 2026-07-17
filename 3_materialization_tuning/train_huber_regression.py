import json
from utility.analyze import build_dataset, fit_huber_dictreg
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

dbt_result_folder = "useful_files/tuva_dbt_run_official_history/original_1216_2321"
MANIFEST_PATH = f"{dbt_result_folder}/manifest.json"
RUN_RESULTS_PATH = f"{dbt_result_folder}/run_results.json"

base_filename = os.path.basename(dbt_result_folder)
FEATURES_JSON_FILE_PATH = f"useful_files/tuva_analysis/for_materialization_tuning/extracted_features_for_training/{base_filename}_extracted_features_for_training.json"
OUTPUT_MODELS_BUNDLE_PATH = f"useful_files/tuva_analysis/for_materialization_tuning/trained_models/{base_filename}_trained_models_bundle.joblib"

# ============= handle run_results.json ===============
with open(RUN_RESULTS_PATH) as f:
    run_results = json.load(f)

# ============= handle manifest.json ===============
with open(MANIFEST_PATH) as f:
    manifest = json.load(f)
nodes = manifest.get("nodes", {})

# ============= load features ===============
with open(FEATURES_JSON_FILE_PATH) as f:
    features = json.load(f)
    time_features = features.get("expanded_time_features", {})
    cardinality_features = features.get("expanded_cardinality_features", {})
    table_stats = features.get("table_stats", {})
    
    table_time_features = {}
    table_cardinality_features = {}
    for k, v in table_stats.items():
        table_time_features[k] = time_features.get(k, {})
        table_cardinality_features[k] = cardinality_features.get(k, {})
        
    print("Feature dict sizes:")
    for name, d in [
        ("expanded_time_features", time_features),
        ("expanded_cardinality_features", cardinality_features),
        ("table_stats", table_stats),
        ("table_time_features", table_time_features),
        ("table_cardinality_features", table_cardinality_features),
    ]:
        print(f"  {name}: {len(d)}")


# --------------- Build datasets ---------------
# 1 Cardinality model: X = expanded_cardinality, y = rows_affected
Xc_dicts, yc_raw, uids_c = build_dataset(table_cardinality_features, table_stats, "rows_affected")

# 2 Slot time model: X = expanded_time, y = slot_ms
Xt_dicts, yt_raw, uids_t = build_dataset(table_time_features, table_stats, "slot_ms")

# --------------- Fit models ---------------
card_pipe, card_dv = fit_huber_dictreg(Xc_dicts, yc_raw, model_name="Cardinality (rows_affected)")
time_pipe,  time_dv = fit_huber_dictreg(Xt_dicts, yt_raw, model_name="Slot Time (slot_ms)")



bundle = {
    "cardinality": {"dv": card_dv, "pipe": card_pipe},
    "slot_time":   {"dv": time_dv, "pipe": time_pipe}
}
# compress levels: 0-9; lz4 is fastest if installed
joblib.dump(bundle, OUTPUT_MODELS_BUNDLE_PATH, compress=("lz4", 3))


'''
def simple_pred_vs_actual(feature_dicts, y_raw, dv, pipe, title="Model fit"):
    # vectorize dict-features with the *trained* DictVectorizer
    X = dv.transform(feature_dicts)

    # model was trained on log1p(target) → back-transform
    y_pred = np.expm1(pipe.predict(X))
    y_true = np.asarray(y_raw, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))

    # consistent axes + 45° reference line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())

    plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title}\nMAE: {mae:,.2f} (original units)")
    # plt.show()
    # save figure
    plt.savefig(f"useful_files/tuva_analysis/0826_simple_pred_vs_actual_figs/{title.replace(' ', '_').lower()}.png")

# Cardinality (rows_affected)
simple_pred_vs_actual(Xc_dicts, yc_raw, card_dv, card_pipe,
                      title="Cardinality")

# Slot time (slot_ms)
simple_pred_vs_actual(Xt_dicts, yt_raw, time_dv, time_pipe,
                      title="Slot_time")

# Bytes scanned (bytes_processed)
simple_pred_vs_actual(Xb_dicts, yb_raw, bytes_dv, bytes_pipe,
                      title="Bytes_processed")
                      
'''