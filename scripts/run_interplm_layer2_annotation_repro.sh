#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERPLM_DIR="${INTERPLM_DIR:-/tmp/InterPLM}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/outputs/interplm_layer2_readme_repro}"
HF_HOME_DIR="${HF_HOME_DIR:-${ROOT_DIR}/.hf_cache}"
LAYER="${LAYER:-2}"
GPU_UUID="${GPU_UUID:-GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26}"
DOWNLOAD_UNIPROT="${DOWNLOAD_UNIPROT:-0}"
EXTRACT_MAX_WORKERS="${EXTRACT_MAX_WORKERS:-}"

export PYTHONPATH="${INTERPLM_DIR}:${PYTHONPATH:-}"
export INTERPLM_DATA="${DATA_ROOT}"
export HF_HOME="${HF_HOME_DIR}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv_cache"

RAW_DIR="${INTERPLM_DATA}/raw"
ANNOTATION_DIR="${INTERPLM_DATA}/annotations/uniprotkb/processed"
EMBED_DIR="${INTERPLM_DATA}/analysis_embeddings/esm2_8m/layer_${LAYER}"
MODEL_DIR="${INTERPLM_DATA}/models/interplm_esm2_8m/layer_${LAYER}"
RESULT_DIR="${INTERPLM_DATA}/results/layer_${LAYER}"
LOG_DIR="${INTERPLM_DATA}/logs"
UNIPROT_PATH="${UNIPROT_PATH:-${RAW_DIR}/uniprot_swissprot_reviewed.tsv.gz}"
UNIPROT_URL="https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Clength%2Csequence%2Cec%2Cft_act_site%2Cft_binding%2Ccc_cofactor%2Cft_disulfid%2Cft_carbohyd%2Cft_lipid%2Cft_mod_res%2Cft_signal%2Cft_transit%2Cft_helix%2Cft_turn%2Cft_strand%2Cft_coiled%2Ccc_domain%2Cft_compbias%2Cft_domain%2Cft_motif%2Cft_region%2Cft_zn_fing%2Cxref_alphafolddb&format=tsv&query=%28reviewed%3Atrue%29"

mkdir -p "${RAW_DIR}" "${MODEL_DIR}" "${RESULT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/readme_pipeline.log") 2>&1

download_uniprot() {
    if [[ -f "${UNIPROT_PATH}" ]]; then
        return 0
    fi
    if [[ "${DOWNLOAD_UNIPROT}" != "1" ]]; then
        echo "Missing UniProt input: ${UNIPROT_PATH}" >&2
        echo "Set DOWNLOAD_UNIPROT=1 to fetch the reviewed Swiss-Prot TSV." >&2
        exit 1
    fi
    wget -O "${UNIPROT_PATH}" "${UNIPROT_URL}"
}

ensure_model_dir() {
    if [[ -f "${MODEL_DIR}/ae_normalized.pt" && -f "${MODEL_DIR}/config.yaml" ]]; then
        return 0
    fi

    export MODEL_DIR
    uv run python - <<'PY'
from pathlib import Path
import os
import shutil
from huggingface_hub import hf_hub_download

target = Path(os.environ["MODEL_DIR"])
target.mkdir(parents=True, exist_ok=True)

weights = Path(
    hf_hub_download(
        repo_id="Elana/InterPLM-esm2-8m",
        filename="layer_2/ae_normalized.pt",
    )
)
shutil.copy2(weights, target / "ae_normalized.pt")
shutil.copy2(
    Path("/tmp/InterPLM/interplm/sae/migration/dummy_config_esm2-8m.yaml"),
    target / "config.yaml",
)
PY
}

run_extract_annotations() {
    if [[ -f "${ANNOTATION_DIR}/uniprotkb_aa_concepts_columns.txt" ]]; then
        return 0
    fi

    uv run python -m interplm.analysis.concepts.extract_annotations \
        --input_uniprot_path "${UNIPROT_PATH}" \
        --output_dir "${ANNOTATION_DIR}" \
        --n_shards 8 \
        --min_required_instances 10 \
        ${EXTRACT_MAX_WORKERS:+--max_workers "${EXTRACT_MAX_WORKERS}"} \
        > "${LOG_DIR}/extract_annotations.log" 2>&1
}

run_embed_annotations() {
    if [[ -f "${EMBED_DIR}/shard_0/embeddings.pt" ]]; then
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${GPU_UUID}" uv run python "${INTERPLM_DIR}/scripts/embed_annotations.py" \
        --input_dir "${ANNOTATION_DIR}" \
        --output_dir "${EMBED_DIR}" \
        --embedder_type esm \
        --model_name facebook/esm2_t6_8M_UR50D \
        --layer "${LAYER}" \
        --batch_size 4 \
        > "${LOG_DIR}/embed_annotations.log" 2>&1
}

run_normalize() {
    if [[ -f "${MODEL_DIR}/feature_stats/max.npy" ]]; then
        return 0
    fi

    CUDA_VISIBLE_DEVICES="${GPU_UUID}" uv run python -m interplm.sae.normalize \
        --sae_dir "${MODEL_DIR}" \
        --aa_embds_dir "${EMBED_DIR}" \
        > "${LOG_DIR}/normalize.log" 2>&1
}

run_prepare_eval() {
    if [[ -f "${ANNOTATION_DIR}/valid/metadata.json" && -f "${ANNOTATION_DIR}/test/metadata.json" ]]; then
        return 0
    fi

    uv run python -m interplm.analysis.concepts.prepare_eval_set \
        --valid_shard_range 0 3 \
        --test_shard_range 4 7 \
        --uniprot_dir "${ANNOTATION_DIR}" \
        --min_aa_per_concept 1000 \
        --min_domains_per_concept 25 \
        > "${LOG_DIR}/prepare_eval_set.log" 2>&1
}

run_compare_and_f1() {
    local eval_set="$1"

    CUDA_VISIBLE_DEVICES="${GPU_UUID}" uv run python -m interplm.analysis.concepts.compare_activations \
        --sae_dir "${MODEL_DIR}" \
        --aa_embds_dir "${EMBED_DIR}" \
        --eval_set_dir "${ANNOTATION_DIR}/${eval_set}" \
        --output_dir "${RESULT_DIR}/${eval_set}_counts" \
        > "${LOG_DIR}/compare_${eval_set}.log" 2>&1

    uv run python -m interplm.analysis.concepts.calculate_f1 \
        --eval_res_dir "${RESULT_DIR}/${eval_set}_counts" \
        --eval_set_dir "${ANNOTATION_DIR}/${eval_set}" \
        > "${LOG_DIR}/calculate_f1_${eval_set}.log" 2>&1
}

run_report() {
    uv run python -m interplm.analysis.concepts.report_metrics \
        --valid_path "${RESULT_DIR}/valid_counts/concept_f1_scores.csv" \
        --test_path "${RESULT_DIR}/test_counts/concept_f1_scores.csv" \
        > "${LOG_DIR}/report_metrics.log" 2>&1
}

main() {
    download_uniprot
    ensure_model_dir
    run_extract_annotations
    run_embed_annotations
    run_normalize
    run_prepare_eval
    run_compare_and_f1 valid
    run_compare_and_f1 test
    run_report
}

main "$@"
