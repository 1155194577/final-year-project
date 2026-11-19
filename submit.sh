#!/usr/bin/env bash
# One-command deploy + submit for the CSE Slurm cluster.
# Run from your Mac with the CSE VPN connected.
#
#   ./submit.sh                        # frame features + phone probe
#   ./submit.sh --epochs 20 --lr 0.1   # flags are passed to phone_probe.py
#   ./submit.sh --follow               # submit, wait, print the job output
#   ./submit.sh --fetch                # copy logs/ back
#
# Consumed here (everything else goes to the job script):
#   -f, --follow   wait for the job, then fetch logs/
#   --fetch        copy remote logs/ back to this machine
#   --no-sync      skip rsync, submit what is already on the cluster
#   --no-data      skip the audio/TextGrids/features in the rsync
#   --with-model   also sync data/SpeechTokenizer (465 MB; normally left out)
#   -h, --help     this text
#
# Env overrides: REMOTE (default "linux1"), REMOTE_DIR, SSH_KEY.
#
# Never synced by default (they exceed the ~235 MB cluster quota): the 465 MB
# checkpoint, the 2.8 GB frames.npz, and the local .venv. The job regenerates
# frames on the compute node in /dev/shm instead.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE:-linux1}"
REMOTE_DIR="${REMOTE_DIR:-final-year-project}"
FOLLOW=0
FETCH=0
SYNC=1
DATA=1
MODEL=0
JOB_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -f|--follow) FOLLOW=1; FETCH=1; shift ;;
        --fetch)     FETCH=1; shift ;;
        --no-sync)   SYNC=0; shift ;;
        --no-data)   DATA=0; shift ;;
        --with-model) MODEL=1; shift ;;
        -h|--help)   sed -n '2,/^set -euo pipefail/p' "$0" | sed -n 's/^# \{0,1\}//p'; exit 0 ;;
        *)           JOB_ARGS+=("$1"); shift ;;
    esac
done

# --- 0. find a key ---------------------------------------------------------
KEY="${SSH_KEY:-}"
if [ -z "$KEY" ]; then
    for candidate in "$HERE/.ssh/id_ed25519" "$HOME/.ssh/id_ed25519" \
                     "$HERE/../slurm/.ssh/id_ed25519"; do
        if [ -f "$candidate" ]; then KEY="$candidate"; break; fi
    done
fi
if [ -z "$KEY" ] || [ ! -f "$KEY" ]; then
    echo "no ssh key: set SSH_KEY=/path/to/key, or put it in .ssh/id_ed25519" >&2
    exit 1
fi

SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 -o BatchMode=yes)

# --- 1. reachable? ---------------------------------------------------------
if ! ssh "${SSH_OPTS[@]}" "$REMOTE" 'true' 2>/dev/null; then
    echo "cannot ssh to $REMOTE - connect to the CSE VPN first (or set REMOTE=...)" >&2
    exit 1
fi

# --- 2. sync -----------------------------------------------------------------
if [ "$SYNC" -eq 1 ]; then
    echo "syncing $HERE/ -> $REMOTE:$REMOTE_DIR/"
    EXCLUDES=(--exclude .git --exclude .ssh --exclude __pycache__ --exclude '*.pyc'
              --exclude '.DS_Store' --exclude '.venv' --exclude logs
              --exclude 'slurm-*.out')
    # 2.9 GB frame tensor: regenerated on the compute node, never shipped.
    # features.npz (21 MB) and phones.npz (64 KB) are still synced.
    EXCLUDES+=(--exclude 'frames.npz')
    [ "$DATA" -eq 0 ] && EXCLUDES+=(--exclude data/)
    [ "$MODEL" -eq 0 ] && EXCLUDES+=(--exclude data/SpeechTokenizer/)
    [ "${ALIGNED:-1}" -eq 0 ] && EXCLUDES+=(--exclude aligned/)
    rsync -az -e "ssh ${SSH_OPTS[*]}" "${EXCLUDES[@]}" "$HERE"/ "$REMOTE:$REMOTE_DIR/"
else
    echo "skipping rsync (--no-sync)"
fi

# --- 3. submit ---------------------------------------------------------------
ARGSTR=""
for arg in ${JOB_ARGS[@]+"${JOB_ARGS[@]}"}; do
    ARGSTR="$ARGSTR$(printf ' %q' "$arg")"
done

JID=$(ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cd '$REMOTE_DIR' && mkdir -p logs && sbatch --parsable gpu.slurm$ARGSTR")

case "$JID" in
    ''|*[!0-9]*) echo "sbatch failed: $JID" >&2; exit 1 ;;
esac

OUT="logs/slurm-$JID.out"
echo "submitted job $JID"
echo "  queue : ssh $REMOTE 'squeue -j $JID'"
echo "  output: ssh $REMOTE 'cd $REMOTE_DIR && tail -f $OUT'"

# --- 4. follow ---------------------------------------------------------------
if [ "$FOLLOW" -eq 1 ]; then
    echo "waiting for job $JID ..."
    while ssh "${SSH_OPTS[@]}" "$REMOTE" "squeue -h -j $JID 2>/dev/null | grep -q ."; do
        sleep 15
    done
    echo "--- $OUT ---"
    ssh "${SSH_OPTS[@]}" "$REMOTE" "cd '$REMOTE_DIR' && tail -n 80 '$OUT'"
fi

# --- 5. fetch -----------------------------------------------------------------
if [ "$FETCH" -eq 1 ]; then
    mkdir -p "$HERE/logs"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$REMOTE:$REMOTE_DIR/logs/" "$HERE/logs/"
    ls -1t "$HERE"/logs/*.out 2>/dev/null | head -n 1
fi