import csv
import json

import numpy as np

from prospero.probe_benchmark.interplm_annotations import (
    build_annotation_report,
    parse_interplm_layer,
)


def test_parse_interplm_layer_handles_interplm_configs():
    assert parse_interplm_layer("interplm_l1_ridge") == 1
    assert parse_interplm_layer("interplm_l6_elasticnet") == 6
    assert parse_interplm_layer("one_hot_ridge") is None


def test_build_annotation_report_maps_coefficients_to_concepts(tmp_path):
    coefficient_dir = tmp_path / "coefficients_run"
    coefficient_dir.mkdir(parents=True, exist_ok=True)

    coefficient_index_path = coefficient_dir / "coefficient_index.csv"
    with open(coefficient_index_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "config",
                "feature_name",
                "estimator_name",
                "budget",
                "seed",
                "selected_hyperparameter",
                "selected_search_value",
                "n_train_samples",
                "n_features",
                "intercept",
                "nnz_coefficients",
                "l1_norm",
                "l2_norm",
                "coefficient_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task": "LGK",
                "config": "interplm_l1_ridge",
                "feature_name": "interplm_l1_mean_pool",
                "estimator_name": "ridge",
                "budget": "8",
                "seed": "1",
                "selected_hyperparameter": "1.0",
                "selected_search_value": "1.0",
                "n_train_samples": "8",
                "n_features": "4",
                "intercept": "0.0",
                "nnz_coefficients": "4",
                "l1_norm": "1.7",
                "l2_norm": "1.1",
                "coefficient_path": "coefficients/LGK/interplm_l1_ridge/budget_8/seed_1.npz",
            }
        )

    artifact_path = coefficient_dir / "coefficients" / "LGK" / "interplm_l1_ridge" / "budget_8" / "seed_1.npz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        coefficients=np.array([0.9, -0.5, 0.1, -0.2], dtype=np.float32),
        intercept=np.array([0.0], dtype=np.float64),
        model_coefficients=np.array([0.9, -0.5, 0.1, -0.2], dtype=np.float32),
        model_intercept=np.array([0.0], dtype=np.float64),
        train_indices=np.array([0, 1, 2], dtype=np.int64),
    )

    dashboard_cache = tmp_path / "dashboard_cache" / "layer_1"
    dashboard_cache.mkdir(parents=True, exist_ok=True)
    with open(dashboard_cache / "Sig_concepts_per_feature.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["feature", "concept", "description", "f1_per_domain", "precision", "recall", "threshold_pct"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "feature": "0",
                "concept": "helix",
                "description": "alpha helix",
                "f1_per_domain": "0.73",
                "precision": "0.80",
                "recall": "0.67",
                "threshold_pct": "0.5",
            }
        )
        writer.writerow(
            {
                "feature": "1",
                "concept": "binding_site",
                "description": "binding site",
                "f1_per_domain": "0.61",
                "precision": "0.70",
                "recall": "0.55",
                "threshold_pct": "0.6",
            }
        )

    output_dir = tmp_path / "mapped"
    manifest = build_annotation_report(
        coefficient_dir=coefficient_dir,
        output_dir=output_dir,
        tasks=("LGK",),
        configs=None,
        budgets=None,
        seeds=None,
        annotation_template=str(tmp_path / "dashboard_cache" / "layer_{layer}" / "Sig_concepts_per_feature.csv"),
        top_k=1,
        coefficient_threshold=0.0,
        max_concepts_per_feature=2,
        include_all_nonzero=False,
    )

    assert manifest["n_coefficient_runs"] == 1
    assert manifest["n_annotated_feature_rows"] == 2

    per_run_path = output_dir / "annotated_interplm_coefficients.csv"
    with open(per_run_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [int(row["feature_id"]) for row in rows] == [0, 1]
    assert rows[0]["concept_names"] == "helix"
    assert rows[1]["concept_names"] == "binding_site"

    aggregated_path = output_dir / "annotated_interplm_coefficients_aggregated.csv"
    with open(aggregated_path, "r", encoding="utf-8", newline="") as handle:
        aggregated_rows = list(csv.DictReader(handle))
    assert len(aggregated_rows) == 2

    with open(output_dir / "annotation_manifest.json", "r", encoding="utf-8") as handle:
        saved_manifest = json.load(handle)
    assert saved_manifest["outputs"]["per_run"] == str(per_run_path)
