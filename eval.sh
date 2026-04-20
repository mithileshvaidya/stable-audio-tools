#!/bin/bash
set -e

CONFIG_DIR="stable_audio_tools/configs/model_configs/autoencoders"
DATASET_CONFIG="stable_audio_tools/configs/dataset_configs/local_daps_test.json"
BATCH_SIZE=8
OUT_PATH="output"
CODECS_DIR="/data/mithilesh/vae_paper/codecs_v1"

NAMES=(
    "vae1_baseline"
    "vae1_powerchannel"
    "vae2_baseline"
    "vae2_powerchannel"
    "vae1_pc_v1_weight_0p5"
    "vae1_pc_v1_weight_0p75"
)
MODEL_CONFIGS=(
    "${CODECS_DIR}/vae1_baseline_config.json"
    "${CODECS_DIR}/vae1_powerchannel_config.json"
    "${CODECS_DIR}/vae2_baseline_config.json"
    "${CODECS_DIR}/vae2_powerchannel_config.json"
    "${CODECS_DIR}/vae1_pc_v1_weight_0p5_config.json"
    "${CODECS_DIR}/vae1_pc_v1_weight_0p75_config.json"
)
CKPTS=(
    "${CODECS_DIR}/vae1_baseline.ckpt"
    "${CODECS_DIR}/vae1_powerchannel.ckpt"
    "${CODECS_DIR}/vae2_baseline.ckpt"
    "${CODECS_DIR}/vae2_powerchannel.ckpt"
    "${CODECS_DIR}/vae1_pc_v1_weight_0p5.ckpt"
    "${CODECS_DIR}/vae1_pc_v1_weight_0p75.ckpt"
)

# One GPU per model, run all in parallel.
# Override GPU_IDS env var to choose which GPUs to use.
GPU_IDS_DEFAULT=(0 1 2 3 4 5 6 7)
if [ -n "${GPU_IDS:-}" ]; then
    IFS=',' read -r -a GPU_IDS_ARR <<< "${GPU_IDS}"
else
    GPU_IDS_ARR=("${GPU_IDS_DEFAULT[@]}")
fi

if [ "${#GPU_IDS_ARR[@]}" -lt "${#NAMES[@]}" ]; then
    echo "Error: need at least ${#NAMES[@]} GPUs (one per model), got ${#GPU_IDS_ARR[@]}: ${GPU_IDS_ARR[*]}" >&2
    exit 1
fi

LOG_FILES=()
PIDS=()

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    cfg="${MODEL_CONFIGS[$i]}"
    ckpt="${CKPTS[$i]}"
    gpu="${GPU_IDS_ARR[$i]}"
    ckpt_dir="$(dirname "${ckpt}")"
    log_file="${ckpt_dir}/${name}_visqol.log"
    LOG_FILES+=("${log_file}|${name}")

    echo "Launching ${name} on GPU ${gpu} -> ${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" python evaluate_autoencoder.py \
        --model_config "${cfg}" \
        --ckpt_path "${ckpt}" \
        --dataset_config ${DATASET_CONFIG} \
        --out_path ${OUT_PATH}/${name} \
        --batch_size ${BATCH_SIZE} \
        --device cuda \
        > "${log_file}" 2>&1 &
    PIDS+=($!)
done

echo
echo "=========================================="
echo "All ${#NAMES[@]} jobs launched in parallel."
echo "Tail any log to monitor, e.g.:  tail -f ${LOG_FILES[0]%|*}"
echo "=========================================="

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "${pid}"; then
        echo "[ok]   ${name} (pid ${pid})"
    else
        rc=$?
        echo "[FAIL] ${name} (pid ${pid}) exit=${rc}"
        FAILED=$((FAILED+1))
    fi
done

echo
echo "=========================================="
echo "Summary (parsed from per-model logs)"
echo "=========================================="

python - <<'PY' "${LOG_FILES[@]}"
import re, sys, os

entries = sys.argv[1:]
rows = []
mos_re = re.compile(r"Mean MOS-LQO\s*:\s*([\-+0-9.]+)\s*[^\d\-+]+\s*([\-+0-9.]+)")
gdb_re = re.compile(r"Mean Gain dB\s*:\s*([\-+0-9.]+)\s*[^\d\-+]+\s*([\-+0-9.]+)")

for entry in entries:
    log, name = entry.split("|", 1)
    text = open(log).read() if os.path.exists(log) else ""
    mos = mos_re.findall(text)
    gdb = gdb_re.findall(text)

    def fmt_mos(p, i):
        return f"{float(p[i][0]):.4f} ± {float(p[i][1]):.4f}" if len(p) > i else "n/a"

    def fmt_db(p, i):
        return f"{float(p[i][0]):+.4f} ± {float(p[i][1]):.4f}" if len(p) > i else "n/a"

    rows.append((
        name,
        fmt_mos(mos, 0),
        fmt_mos(mos, 1),
        fmt_db(gdb, 0),
        fmt_db(gdb, 1),
    ))

headers = ("Model", "Recon MOS-LQO", "Swapped MOS-LQO", "Recon Gain dB", "Swapped Gain dB")
widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
fmt = " | ".join(f"{{:<{w}}}" for w in widths)
sep = "-+-".join("-" * w for w in widths)

print(fmt.format(*headers))
print(sep)
for r in rows:
    print(fmt.format(*r))
PY

if [ "${FAILED}" -gt 0 ]; then
    echo
    echo "WARNING: ${FAILED} job(s) failed. Inspect their logs above."
    exit 1
fi

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
