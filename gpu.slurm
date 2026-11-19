#!/bin/bash
#SBATCH -J fyp-phone
#SBATCH -o logs/slurm-%j.out
#SBATCH -p gpu_2h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:45:00
#
#   sbatch gpu.slurm                        # extract frame features, then probe
#   sbatch gpu.slurm --epochs 20 --lr 0.1   # probe flags are passed through
#   FRAMES=/dev/shm/frames_123 sbatch gpu.slurm   # reuse a cached frames.npz
#
# Phone probing needs one vector per 25 fps frame, but frames.npz is 2.8 GB and
# the home quota has ~235 MB free. So the frame features are extracted on the
# compute node into /dev/shm and linked into <dataset>/features/ for the probe;
# they are deleted when the job ends unless KEEP_SHM=1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this from the repo directory}"
mkdir -p logs

IMAGE=${SINGULARITY_IMAGE:-/dev/shm/pytorch-2.5.1.sif}
DATA=data/libritts_preprocessed/dev.clean_36speakers_20utts
FEATURES_DIR=${FRAMES:-/dev/shm/frames_$SLURM_JOB_ID}

if [[ ! -f "$IMAGE" ]]; then
    echo "image missing on $(hostname), pulling it now (~5 min)"
    export SINGULARITY_CACHEDIR=/dev/shm/sgc SINGULARITY_TMPDIR=/dev/shm/stmp
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
    ( cd /dev/shm && singularity pull --name pytorch-2.5.1.sif \
        docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime )
fi

echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

DEPS=/dev/shm/deps_$SLURM_JOB_ID
MODEL_DIR=/dev/shm/SpeechTokenizer_$SLURM_JOB_ID
cleanup() {
    rm -f "$DATA/features/frames.npz"        # the symlink, never the target
    if [[ "${KEEP_SHM:-0}" != "1" ]]; then
        rm -rf "$DEPS" "$MODEL_DIR" "$FEATURES_DIR"
    else
        echo "kept $FEATURES_DIR (pass FRAMES=$FEATURES_DIR to reuse)"
    fi
}
trap cleanup EXIT

singularity exec --nv "$IMAGE" python -m pip install -q --no-cache-dir --target "$DEPS" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    speechtokenizer beartype soundfile pyarrow tensorboard

run() { singularity exec --nv --env PYTHONPATH="$DEPS" "$IMAGE" python -u "$@"; }

if [[ ! -f "$FEATURES_DIR/frames.npz" ]]; then
    echo "== frame-level features -> $FEATURES_DIR =="
    run speechtokenizer_features.py \
        --dataset-dir "$DATA" --device cuda --model-dir "$MODEL_DIR" \
        --frames --output-dir "$FEATURES_DIR"
else
    echo "== reusing $FEATURES_DIR/frames.npz =="
fi

# The probe reads <dataset>/features/{frames,phones,features}.npz. Only the
# 2.8 GB frame tensor lives off-quota; the other two are already in the repo.
mkdir -p "$DATA/features"
ln -sfn "$FEATURES_DIR/frames.npz" "$DATA/features/frames.npz"

echo "== phone probe =="
run phone_probe.py --data-dir "$DATA" --device cuda "$@"