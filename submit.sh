#!/usr/bin/env bash
# Deploy + submit to the CSE GPGPU cluster.
# Run from your Mac with the CSE VPN connected.
#
#   ./submit.sh                                  # sync code+data, run the phone probe
#   ./submit.sh --follow                          # ... and wait for it, then fetch logs
#   ./submit.sh probe --epochs 20                 # extra args go to phone_probe.py
#   ./submit.sh train affine_vc.py --male 174     # arbitrary python command
#   ./submit.sh --node gpu7 --no-data             # a different node, skip the data sync
#   ./submit.sh --dry-run                         # print what would run, do nothing
#
# Everything here that touches slurm must export SLURM_CONF first - the cluster
# has no /etc/slurm/slurm.conf, and forgetting it gives:
#   sbatch: error: s_p_parse_file: unable to status file /etc/slurm/slurm.conf
#
# Env overrides: REMOTE (linux1), REMOTE_DIR (~/final-year-project),
#                WORKDIR (/research/d7/fyp26/cyho2), SLURM_CONF, PARTITION, QOS,
#                ACCOUNT, CPUS, TIMELIMIT, NODE, SSH_KEY.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE:-linux1}"
REMOTE_DIR="${REMOTE_DIR:-final-year-project}"
WORKDIR="${WORKDIR:-/research/d7/fyp26/cyho2}"

SLURM_CONF="${SLURM_CONF:-/opt1/slurm/gpu-slurm.conf}"
PARTITION="${PARTITION:-gpu_2h}"
QOS="${QOS:-gpu}"
ACCOUNT="${ACCOUNT:-gpu}"
CPUS="${CPUS:-6}"
TIMELIMIT="${TIMELIMIT:-01:45:00}"
NODE="${NODE:-projgpu33}"      # A6000 48GB + mounts /research/d7; gpu7/gpu9 also work

FOLLOW=0
FETCH=0
SYNC=1
DATA=1
MODEL=1
DRY=0
AFTER=""
MODE="probe"
JOB_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--follow) FOLLOW=1; FETCH=1; shift ;;
        --fetch)     FETCH=1; shift ;;
        --no-sync)   SYNC=0; shift ;;
        --no-data)   DATA=0; shift ;;
        --no-model)  MODEL=0; shift ;;
        --dry-run)   DRY=1; SYNC=0; shift ;;
        --node)      NODE="$2"; shift 2 ;;
        --after)     AFTER="$2"; shift 2 ;;
        --workdir)   WORKDIR="$2"; shift 2 ;;
        -h|--help)   sed -n '2,/^set -euo pipefail/p' "$0" | sed -n 's/^# \{0,1\}//p'; exit 0 ;;
        probe|frames|train) MODE="$1"; shift ;;
        *)           JOB_ARGS+=("$1"); shift ;;
    esac
done

DATASET=dev.clean_36speakers_20utts
DATA_LOCAL="$HERE/data/libritts_preprocessed/$DATASET"

# --- 0. ssh key -------------------------------------------------------------
KEY="${SSH_KEY:-}"
if [[ -z "$KEY" ]]; then
    for candidate in "$HERE/.ssh/id_ed25519" "$HOME/.ssh/id_ed25519"; do
        [[ -f "$candidate" ]] && { KEY="$candidate"; break; }
    done
fi
[[ -n "$KEY" && -f "$KEY" ]] || { echo "no ssh key: set SSH_KEY=..." >&2; exit 1; }
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 -o BatchMode=yes)

ssh_do() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }
# every slurm call needs the conf exported on the remote side
ssh_slurm() { ssh_do "export SLURM_CONF='$SLURM_CONF'; $1"; }

# --- 1. reachable? ----------------------------------------------------------
if ! ssh_do 'true' 2>/dev/null; then
    echo "cannot ssh to $REMOTE - connect to the CSE VPN first (or set REMOTE=...)" >&2
    exit 1
fi
if ! ssh_slurm 'sinfo -p '"$PARTITION"' >/dev/null 2>&1'; then
    echo "SLURM_CONF=$SLURM_CONF does not give a usable '$PARTITION' partition" >&2
    exit 1
fi

if [[ "$DRY" -eq 1 ]]; then
    echo "dry run: would sync $HERE -> $REMOTE:$REMOTE_DIR and $DATA_LOCAL -> $WORKDIR/datasets/$DATASET"
    echo "dry run: would submit '$MODE' on ${NODE:-any node} with ${JOB_ARGS[*]:-no extra args}"
    exit 0
fi

# --- 2. queue guard ---------------------------------------------------------
PENDING=$(ssh_slurm 'squeue -h -u $USER -t PENDING 2>/dev/null | wc -l' | tr -d ' ')
if [[ "${PENDING:-0}" -ge 4 ]]; then
    echo "warning: $PENDING jobs already pending; submissions may be rejected" >&2
    echo "         (a per-user pending cap around 4 was observed on 2026-11-22)" >&2
fi

# --- 3. sync ----------------------------------------------------------------
if [[ "$SYNC" -eq 1 ]]; then
    echo "== code -> $REMOTE:$REMOTE_DIR/"
    EXCLUDES=(--exclude .git --exclude .ssh --exclude __pycache__ --exclude '*.pyc'
              --exclude .DS_Store --exclude .venv --exclude logs --exclude 'slurm-*.out'
              --exclude 'data/' --exclude third_party/ --exclude '*.pt')
    ssh_do "mkdir -p '$REMOTE_DIR' '$WORKDIR/datasets' '$WORKDIR/features' '$WORKDIR/checkpoints' '$WORKDIR/logs'"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${EXCLUDES[@]}" "$HERE/" "$REMOTE:$REMOTE_DIR/"

    if [[ "$DATA" -eq 1 ]]; then
        echo "== data -> $REMOTE:$WORKDIR/datasets/$DATASET/"
        # frames.npz is 2.8 GB and regenerated on the compute node; never shipped
        rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude 'frames.npz' \
            "$DATA_LOCAL/" "$REMOTE:$WORKDIR/datasets/$DATASET/"
    fi
    if [[ "$MODEL" -eq 1 && -d "$HERE/data/SpeechTokenizer" ]]; then
        echo "== model -> $REMOTE:$WORKDIR/datasets/SpeechTokenizer/"
        rsync -az -e "ssh ${SSH_OPTS[*]}" \
            "$HERE/data/SpeechTokenizer/" "$REMOTE:$WORKDIR/datasets/SpeechTokenizer/"
    fi
else
    echo "== skipping sync (--no-sync)"
fi

# --- 4. submit --------------------------------------------------------------
ARGS="$MODE"
for a in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do ARGS="$ARGS$(printf ' %q' "$a")"; done

NODE_FLAG=""
[[ -n "$NODE" ]] && NODE_FLAG="-w $NODE"
DEP_FLAG=""
[[ -n "$AFTER" ]] && DEP_FLAG="--dependency=afterok:$AFTER"

SUBMIT="cd '$REMOTE_DIR' && mkdir -p logs && sbatch --parsable \
 -p $PARTITION --qos=$QOS --account=$ACCOUNT --gres=gpu:1 -c $CPUS -t $TIMELIMIT \
 $NODE_FLAG $DEP_FLAG --export=ALL,WORKDIR='$WORKDIR' gpu.slurm $ARGS"

if [[ "$DRY" -eq 1 ]]; then
    echo "--- dry run, would execute:"
    echo "ssh $REMOTE \"export SLURM_CONF='$SLURM_CONF'; $SUBMIT\""
    exit 0
fi

echo "== submitting ($MODE on ${NODE:-any node})"
JID=$(ssh_slurm "$SUBMIT")
case "$JID" in
    ''|*[!0-9]*) echo "sbatch failed: $JID" >&2; exit 1 ;;
esac

OUT="$WORKDIR/logs/slurm-$JID.out"
echo "submitted job $JID"
echo "  queue : ssh $REMOTE \"export SLURM_CONF=$SLURM_CONF; squeue -j $JID\""
echo "  output: ssh $REMOTE 'tail -f $OUT'"

# --- 5. follow --------------------------------------------------------------
if [[ "$FOLLOW" -eq 1 ]]; then
    echo "== waiting for $JID ..."
    while [[ -n "$(ssh_slurm "squeue -h -j $JID 2>/dev/null" || true)" ]]; do sleep 20; done
    echo "--- $OUT ---"
    ssh_do "tail -n 80 '$OUT' 2>/dev/null || echo '(no output yet)'"
fi

# --- 6. fetch ---------------------------------------------------------------
if [[ "$FETCH" -eq 1 ]]; then
    mkdir -p "$HERE/logs"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE:$WORKDIR/logs/" "$HERE/logs/" || true
fi
