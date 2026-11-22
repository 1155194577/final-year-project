#!/usr/bin/env bash
# Push the two large assets the cluster cannot fetch itself (no egress):
#   - the SpeechTokenizer checkpoint (482 MB)
#   - the offline wheelhouse (118 MB)
#
# Resumable: each file goes up in 32 MB chunks written with dd + seek, so a
# dropped VPN connection costs one chunk, not the whole transfer. Re-running
# this script is safe; it verifies sizes at the end.
#
#   ./push_assets.sh            # push whatever is missing
#   ./push_assets.sh --verify   # only report remote sizes
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="${REMOTE:-linux1}"
WORKDIR="${WORKDIR:-/research/d7/fyp26/cyho2}"
KEY="${SSH_KEY:-$HERE/.ssh/id_ed25519}"
CHUNK_MB=${CHUNK_MB:-32}
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o ConnectTimeout=10 -o BatchMode=yes
          -o ServerAliveInterval=20 -o ServerAliveCountMax=6)

ASSETS=(
  "$HERE/data/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
  "$HERE/wheelhouse.tar"
)
DEST_DIR="$WORKDIR/assets"

ssh_do() { ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"; }

ssh_do "mkdir -p '$DEST_DIR'"

tar_needed=0
if [[ -d "$HERE/wheelhouse" && ! -f "$HERE/wheelhouse.tar" ]]; then
    tar_needed=1
fi

if [[ "${1:-}" == "--verify" ]]; then
    for a in "${ASSETS[@]}"; do
        [[ -f "$a" ]] || continue
        local_size=$(stat -f%z "$a")
        remote_size=$(ssh_do "stat -c%s '$DEST_DIR/$(basename "$a")' 2>/dev/null || echo 0" | tr -d ' ')
        echo "$(basename "$a"): local $local_size | remote $remote_size $([[ "$local_size" == "$remote_size" ]] && echo OK || echo MISMATCH)"
    done
    exit 0
fi

if [[ "$tar_needed" -eq 1 ]]; then
    echo "packing wheelhouse ..."
    tar cf "$HERE/wheelhouse.tar" -C "$HERE" wheelhouse
fi

for src in "${ASSETS[@]}"; do
    [[ -f "$src" ]] || { echo "missing local asset: $src" >&2; continue; }
    name=$(basename "$src")
    dst="$DEST_DIR/$name"
    size=$(stat -f%z "$src")
    chunk=$((CHUNK_MB*1024*1024))
    total=$(( (size + chunk - 1) / chunk ))

    remote_size=$(ssh_do "stat -c%s '$dst' 2>/dev/null || echo 0" | tr -d ' ')
    if [[ "$remote_size" == "$size" ]]; then
        echo "$name: already complete ($size bytes), skipping"
        continue
    fi

    echo "$name: $((size/1048576)) MB in $total chunks (remote has $((remote_size/1048576)) MB)"
    for ((i=0; i<total; i++)); do
        start=$((i*chunk))
        # skip chunks already written
        if (( start + chunk <= remote_size )); then continue; fi
        ok=0
        for attempt in 1 2 3 4 5 6; do
            if dd if="$src" bs=1048576 skip=$((start/1048576)) count=$CHUNK_MB 2>/dev/null \
               | ssh "${SSH_OPTS[@]}" "$REMOTE" \
                   "dd of='$dst' bs=1048576 seek=$((start/1048576)) conv=notrunc 2>/dev/null"; then
                ok=1; break
            fi
            echo "  chunk $((i+1))/$total failed (try $attempt), retry in 5s ..." >&2
            sleep 5
        done
        if [[ "$ok" -ne 1 ]]; then
            echo "  giving up on chunk $((i+1)); rerun to resume" >&2
            exit 1
        fi
        printf '  %d/%d chunks\r' $((i+1)) "$total"
    done
    echo ""

    remote_size=$(ssh_do "stat -c%s '$dst' 2>/dev/null || echo 0" | tr -d ' ')
    if [[ "$remote_size" == "$size" ]]; then
        echo "$name: OK ($size bytes)"
    else
        echo "$name: SIZE MISMATCH (local $size, remote $remote_size)" >&2
        exit 1
    fi
done

# unpack the wheelhouse next to the assets
ssh_do "cd '$DEST_DIR' && tar xf wheelhouse.tar && ls wheelhouse | wc -l"
echo "done"
