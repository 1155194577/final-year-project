# SLURM on the CSE cluster — working notes

Measured on **2026-11-22** from `linux1` (`cyho2@linux1`). The cluster changes
under you: images disappear, nodes go unavailable. Re-check before trusting any
specific node name.

Context: the goal was to run SpeechTokenizer probing / VC experiments with a
PyTorch container on `gpu_2h`. The data and checkpoint are on the cluster; the
blocker is item 5 (the container image).

---

## 0. The one thing that breaks every first command

There is no `/etc/slurm/slurm.conf`. **Every** `sbatch`/`sinfo`/`squeue` needs
`SLURM_CONF` exported, or you get:

```
sbatch: error: s_p_parse_file: unable to status file /etc/slurm/slurm.conf
         No such file or directory, retrying in 1sec up to 60sec
```

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf   # GPGPU partition
export SLURM_CONF=/opt1/slurm/hpc-slurm.conf   # HPC (CPU) partition
```

`submit.sh` and every template below already do this. For interactive shells,
add aliases to `~/.bashrc`.

---

## 1. Partitions

| partition | QOS | account | purpose |
|---|---|---|---|
| `gpu_2h` | `gpu` | `gpu` | GPGPU, 1 GPU/job, **2 h wall** |
| `hpc_72h` | `hpc` | — | CPU only, 192 cores/node, 72 h wall |

HPC group cap: **80 CPUs across all users**. `QOSGrpCpuLimit` while pending
means reduce `-c` (16 → 8 → 4).

---

## 2. `gpu_2h` is heterogeneous

`sinfo` on 2026-11-22:

| nodes | GPU | VRAM | notes |
|---|---|---|---|
| `gpu1 gpu3 gpu24 gpu27 gpu28 gpu29` | RTX 3090 | 24 GB | Ampere, tensor cores |
| `gpu54 gpu55` | Titan RTX | 24 GB | Turing |
| `projgpu33` | RTX 3090 ×3 + **A6000** | 24 / **48 GB** | `/research/d7` mounted |
| `gpu7 gpu8` | Titan X | 12 GB | oldest, no tensor cores |
| `gpu9` | Titan X | 12 GB | fine for `/research/d7` |

`sinfo` can say `mix` about a node that is unusable: `gpu28` returned
`ReqNodeNotAvail` while listed. Check with a 2-minute probe (item 10), not `sinfo`.

---

## 3. Storage — and what is actually writable

| path | size | notes |
|---|---|---|
| `/uac/y22/cyho2` (home) | quota **2442 M**, ~1248 M used | ~1.2 GB free; **not** for datasets/checkpoints |
| `/research/d7/fyp26` | 140 TB (86 TB used) | **root-owned, mode 755 → NOT writable** |
| `/research/d7/fyp26/cyho2` | 140 TB | **writable** — the actual project store |
| `/dev/shm` (per node) | RAM (126 GB on projgpu33) | **volatile, see item 5** |

Verified layout under `/research/d7/fyp26/cyho2/`:

```
datasets/     # synced dataset + SpeechTokenizer checkpoint
features/     # frame features (frames.npz), shared across jobs
checkpoints/  # training checkpoints
logs/         # slurm-%j.out
assets/       # wheelhouse (+ wheelhouse.tar)
images/       # mfa.sif (no torch)
```

`touch /research/d7/fyp26/x` → `Permission denied`; `touch .../cyho2/x` succeeds.
Everything in `submit.sh`/`gpu.slurm` defaults to `WORKDIR=/research/d7/fyp26/cyho2`.

`/research/d7` mount status per node: `projgpu33` ✔, `gpu7`/`gpu9` ✔, `gpu8`
autofs hangs, `gpu36 gpu56 gpu57 gpu59` cannot resolve it.

---

## 4. The cluster has NO internet (verified)

Both the login node and the compute nodes are isolated:

```
linux1:      pip download speechtokenizer → ERROR: No matching distribution found
             curl https://huggingface.co    → no response
projgpu33:   PYPI_BLOCKED
             HF_BLOCKED
```

Consequences — every one of these is now a design constraint:

1. `pip install` inside a job **cannot** work → ship a wheelhouse (item 7).
2. The model checkpoint cannot be downloaded on the node → ship it (item 8).
3. `singularity pull` cannot work → the `.sif` must already be on the node (item 5).
4. `git clone`/`git pull` cannot work → code goes by `rsync` from the Mac.

Do **not** rely on any job-time download.

---

## 5. The container image is volatile — this is the current blocker

`gpu.slurm` needs a PyTorch container at `/dev/shm/pytorch-2.5.1.sif`
(python 3.11, torch 2.5.1+cu121, torchaudio 2.5.1, numpy 2.1.2).

Observed on 2026-11-22 — the same node lost it mid-session:

| time | node | `/dev/shm` image state |
|---|---|---|
| 21:00 | projgpu33 | present, `singularity exec` works (python 3.11.10) |
| 22:03 | projgpu33 | **missing**; pull fails: `network is unreachable` |
| 22:08 | gpu7 | present, 3.0 GB, dated Nov 4; `du -sh /dev/shm` = 13 G |
| 22:12 | gpu7 | present, deps install begins |
| 22:15 | gpu7 | **missing**; pull fails on `no such file or directory` |
| 22:22 | gpu7 | `/dev/shm` = 24 K, empty |

Also checked: `gpu8` never had it; `~/.singularity` cache is 56 K (nothing to
rebuild from); the only image in project storage is `images/mfa.sif` (MFA, no
torch). So **nothing on the cluster can run torch right now**, and it cannot be
re-created without egress.

If the image is missing, the job fails immediately with:

```
FATAL: While making image from oci registry: failed to get checksum ...:
       dial tcp ...: connect: network is unreachable
```

Recovery options, in order of practicality:

1. Ask the cluster admin / whoever maintains the node how the image is meant to
   persist (`/dev/shm` is plain tmpfs; something is cleaning it).
2. Copy a known-good `.sif` up from a machine with internet.
   **Cost warning:** a 3 GB upload at the observed ~0.17 MB/s is ≈ 5 hours.
   Use the chunked method in item 8, not `scp`.
3. Use a different machine with egress (e.g. a rented GPU box) instead.

Do not assume `/dev/shm` contents survive between two of your own jobs.

---

## 6. Singularity does not bind `/research` — pass `--bind`

Default binds are `$HOME`, `/tmp`, `/proc`, `/sys`, `/dev`, and the CWD. Any path
under `/research` is **invisible inside the container**, which looks like "file
not found" even though it exists on the host:

```
WARNING: Location '/research/d7/fyp26/cyho2/assets/wheelhouse' is ignored:
         it is either a non-existing path or lacks a specific scheme.
```

Fix (already in `gpu.slurm`):

```bash
BIND="--bind $WORKDIR:$WORKDIR"
singularity exec --nv $BIND --env PYTHONPATH="$DEPS" "$IMAGE" python -u "$@"
```

---

## 7. Offline Python dependencies (wheelhouse)

The container has torch/torchaudio/numpy but **not** the project deps:
`soundfile pyarrow einops scipy huggingface_hub tensorboard beartype` and
`speechtokenizer` itself.

Build the wheelhouse **on the Mac** for the container's platform:

```bash
cd final-year-project && mkdir -p wheelhouse
PKGS="speechtokenizer beartype einops huggingface_hub lion_pytorch accelerate \
matplotlib scipy soundfile pyarrow tensorboard contourpy cycler fonttools kiwisolver \
pillow pyparsing python-dateutil six certifi charset-normalizer idna urllib3 filelock \
fsspec packaging pyyaml requests tqdm typing-extensions psutil safetensors absl-py \
grpcio markdown protobuf tensorboard-data-server werkzeug cffi pycparser"
.venv/bin/pip download -q --no-deps --only-binary=:all: \
    --platform manylinux2014_x86_64 --python-version 3.11 -d wheelhouse $PKGS
```

Result on 2026-11-22: **40 wheels, 118 MB**.

Install inside the job. Two traps:

```bash
# --no-deps is REQUIRED: with --target pip resolves against the target dir,
# not the image, so it demands numpy/torch wheels the image already has.
#   ERROR: Could not find a version that satisfies the requirement numpy (from speechtokenizer)
singularity exec --nv $BIND "$IMAGE" python -m pip install -q --no-index --no-deps \
    --find-links "$WHEELS" --target "$DEPS" $DEP_PKGS
```

`--no-index` alone is not enough; `--no-deps` + an explicit package list is what
makes an offline install work against a pre-populated image.

---

## 8. Moving large files from macOS — openrsync will bite you

macOS ships **openrsync (protocol 29)**, not GNU rsync. It does **not** support
`--inplace`, `--append-verify` or `--partial` in the usual way; passing them makes
it print its usage text and fail:

```
rsync attempt 3 failed, retrying in 10s ...
[--out-format=FMT] [--partial] [--partial-dir] ...   <- usage dump
```

Use plain `rsync -az` for small trees, and **chunked `dd` over ssh** for big
files — resumable at chunk granularity with no extra disk:

```bash
# 32 MB chunks, seek on the remote side, verify size at the end
dd if="$src" bs=1048576 skip=$((start/1048576)) count=32 \
  | ssh "$REMOTE" "dd of='$dst' bs=1048576 seek=$((start/1048576)) conv=notrunc"
```

`push_assets.sh` implements this (checkpoint + wheelhouse, with size verification
and per-chunk retries). Re-running it is safe: complete files are skipped.

Observed link behaviour: ~0.17 MB/s, and the VPN dropped mid-transfer once after
47 minutes (`Broken pipe` / `Can't assign requested address`). Expect to re-run.

---

## 9. Hard limits (each costs a job if forgotten)

| flag | constraint | failure |
|---|---|---|
| `--mem` | **never pass it** (`RealMemory=1`) | `Requested node configuration is not available` |
| `-c` | ≤ 6 per GPU; 8 needs `--constraint=highcpucount` | same |
| `-t` | ≤ 2 h on `gpu_2h` | killed at the wall |
| `-w a,b,c` | one node per job; multiple hosts → `Node count specification invalid` | rejected |
| pending jobs | ~4 per user (5th onward rejected while 4 pending; observed) | silent failure in a loop |

The 2 h wall is fixed — checkpoint and chain:

```bash
sbatch gpu.slurm            # writes a checkpoint
sbatch --dependency=afterok:$JID gpu.slurm --resume
```

---

## 10. Templates

**Probe a node before trusting it** (2 min, answers everything at once):

```bash
ssh linux1 'export SLURM_CONF=/opt1/slurm/gpu-slurm.conf; \
sbatch --parsable -p gpu_2h --qos=gpu --account=gpu --gres=gpu:1 -c 2 -t 00:03:00 \
  -w gpu7 -o ~/probe-%j.out --wrap "
  hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
  ls /dev/shm/*.sif 2>/dev/null || echo NO_IMAGE;
  ls /research/d7/fyp26/cyho2 >/dev/null 2>&1 && echo DATA_MOUNTED || echo NO_DATA;
  curl -sI -m 5 https://pypi.org >/dev/null 2>&1 && echo NET_OK || echo NET_BLOCKED"'
```

**This repo's submit script** (handles SLURM_CONF, sync, node pinning, queue guard):

```bash
./submit.sh --dry-run              # show what would happen
./submit.sh --node gpu7 check      # environment check, ~1 min
./submit.sh --node gpu7 probe      # frame features + phone probe
./submit.sh --node gpu7 train affine_vc.py --male 174 --female 84
./submit.sh --follow               # wait, print output, fetch logs
```

**CPU batch job (MFA, preprocessing, sweeps):**

```bash
SLURM_CONF=/opt1/slurm/hpc-slurm.conf sbatch -p hpc_72h --qos=hpc -c 16 \
  -t 4:00:00 -o ~/job-%j.out --wrap "cd ~/final-year-project && ./run.sh"
```

**Monitoring:**

```bash
SLURM_CONF=/opt1/slurm/gpu-slurm.conf squeue -u $USER -o "%.8i %.10T %.6C %R"
SLURM_CONF=/opt1/slurm/gpu-slurm.conf sinfo -p gpu_2h -o "%n %t %G %c"
tail -f /research/d7/fyp26/cyho2/logs/slurm-<JID>.out
```

---

## 11. Pre-flight checklist

- [ ] `SLURM_CONF` exported (or use `submit.sh`)
- [ ] no `--mem`; `-c` ≤ 6; `-t` ≤ 2 h
- [ ] one node in `-w`, and it has the **container image** (`ls /dev/shm/*.sif`)
- [ ] that node mounts `/research/d7` if the job needs it
- [ ] wheelhouse present at `$WORKDIR/assets/wheelhouse`
- [ ] dataset + checkpoint present under `$WORKDIR/datasets`
- [ ] fewer than ~4 jobs pending
- [ ] job is idempotent: a rerun must not clobber existing checkpoints

---

## 12. Current state of this project's deployment (2026-11-22 22:30)

Done and verified:

- code at `~/final-year-project` (rsync, no git on the cluster)
- dataset at `$WORKDIR/datasets/dev.clean_36speakers_20utts` — 212 MB
- checkpoint at `$WORKDIR/datasets/SpeechTokenizer/speechtokenizer_hubert_avg/`
  — `SpeechTokenizer.pt` 481,906,997 bytes (matches local) + `config.json`
- wheelhouse at `$WORKDIR/assets/wheelhouse` — 40 wheels, 118 MB
- job script `gpu.slurm` with `probe | frames | train | check` modes, offline
  install, container bind, fail-fast checks

Blocked on: **no PyTorch container image on any accessible node** (item 5).

To finish once the image is back on `gpu7`:

```bash
./submit.sh --no-data --no-model --node gpu7 check   # ~1 min
./submit.sh --no-data --no-model --node gpu7 probe   # then the real work
```

---

## 13. Verdict

Slurm here is fine for **submitting finished, resumable jobs** and poor for
**iterating** — queue waits, a 2 h wall, a ~4-job pending cap, no internet, and
a volatile `/dev/shm` image cache.

- probes / sweeps / MFA → **HPC CPU** partition
- GPU work → `gpu_2h`, pinned to a node that currently has the image
- anything needing `pip`, `git` or model downloads → **do not plan on the cluster**;
  vendor it, or use a machine with egress
