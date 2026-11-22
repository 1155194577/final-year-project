#!/bin/bash
#SBATCH -J fyp
#SBATCH -o /research/d7/fyp26/cyho2/logs/slurm-%j.out
#SBATCH -p gpu_2h
#SBATCH --qos=gpu
#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:45:00
#
# NOTE: no --mem. Nodes report RealMemory=1, so any memory request is rejected
#       with "Requested node configuration is not available".
#
#   sbatch -w projgpu33 gpu.slurm probe --epochs 20
#   sbatch -w projgpu33 gpu.slurm train affine_vc.py --male 174 --female 84
#   sbatch -w projgpu33 gpu.slurm frames
#
# Everything that must persist lives in WORKDIR (default /research/d7/fyp26/cyho2,
# the only writable project storage). /dev/shm is per-node and wiped at job end.
#
# Modes:
#   probe   extract frame-level features if missing, then run phone_probe.py   [default]
#   frames  extract frame-level features only
#   train   run an arbitrary python command, e.g. "train affine_vc.py ..."

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit this from the repo directory}"

WORKDIR=${WORKDIR:-/research/d7/fyp26/cyho2}
DATASETS=${DATASETS:-$WORKDIR/datasets}
DATA=${DATA:-$DATASETS/dev.clean_36speakers_20utts}
CHECKPOINTS=${CHECKPOINTS:-$WORKDIR/checkpoints}
LOGS=${LOGS:-$WORKDIR/logs}
FEATURES_DIR=${FRAMES:-$WORKDIR/features/frames}
IMAGE=${SINGULARITY_IMAGE:-/dev/shm/pytorch-2.5.1.sif}

MODE=${1:-probe}
shift || true

mkdir -p "$CHECKPOINTS" "$LOGS" "$WORKDIR/features"

# Fail fast rather than after the image pull.
touch "$WORKDIR/.wtest" 2>/dev/null || {
    echo "WORKDIR $WORKDIR is not writable from $(hostname)" >&2
    exit 1
}
rm -f "$WORKDIR/.wtest"

echo "job ${SLURM_JOB_ID} on $(hostname), $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "data=$DATA"
echo "features=$FEATURES_DIR  checkpoints=$CHECKPOINTS"

if [[ ! -d "$DATA" ]]; then
    echo "dataset missing at $DATA - run ./submit.sh first to sync it" >&2
    exit 1
fi

if [[ ! -f "$IMAGE" ]]; then
    echo "image missing on $(hostname), pulling it now (~5 min)"
    export SINGULARITY_CACHEDIR=/dev/shm/sgc SINGULARITY_TMPDIR=/dev/shm/stmp
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
    ( cd /dev/shm && singularity pull --name pytorch-2.5.1.sif \
        docker://pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime )
fi

DEPS=/dev/shm/deps_$SLURM_JOB_ID
MODEL_DIR=${MODEL_DIR:-$DATASETS/SpeechTokenizer/speechtokenizer_hubert_avg}
cleanup() {
    rm -f "$DATA/features/frames.npz"        # the symlink, never the target
    [[ "${KEEP_SHM:-0}" == "1" ]] || rm -rf "$DEPS"
}
trap cleanup EXIT

singularity exec --nv "$IMAGE" python -m pip install -q --no-cache-dir --target "$DEPS" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    speechtokenizer beartype soundfile pyarrow tensorboard

run() { singularity exec --nv --env PYTHONPATH="$DEPS" "$IMAGE" python -u "$@"; }

extract_frames() {
    if [[ -f "$FEATURES_DIR/frames.npz" ]]; then
        echo "== reusing $FEATURES_DIR/frames.npz =="
    else
        echo "== extracting frame-level features -> $FEATURES_DIR =="
        run speechtokenizer_features.py \
            --dataset-dir "$DATA" --device cuda --model-dir "$MODEL_DIR" \
            --frames --output-dir "$FEATURES_DIR"
    fi
    # the probe reads <dataset>/features/frames.npz
    mkdir -p "$DATA/features"
    ln -sfn "$FEATURES_DIR/frames.npz" "$DATA/features/frames.npz"
}

case "$MODE" in
    frames) extract_frames ;;
    probe)
        extract_frames
        echo "== phone probe =="
        run phone_probe.py --data-dir "$DATA" --device cuda "$@"
        ;;
    train)
        echo "== train: $* =="
        run "$@"
        ;;
    *)
        echo "unknown mode '$MODE' (use probe|frames|train)" >&2
        exit 2
        ;;
esac
