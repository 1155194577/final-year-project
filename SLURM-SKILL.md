# SLURM on the CSE cluster — working notes

Everything here was measured on 2026-11-22 from `linux1` (`cyho2@linux1`).
Cluster state changes; re-check before relying on a node name.

---

## 0. The one thing that breaks every first command

There is no `/etc/slurm/slurm.conf` on this cluster. **Every** `sbatch`/`sinfo`/`squeue`
needs `SLURM_CONF` exported first, or you get:

```
sbatch: error: s_p_parse_file: unable to status file /etc/slurm/slurm.conf
         No such file or directory, retrying in 1sec up to 60sec
```

```bash
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf   # GPGPU partition
export SLURM_CONF=/opt1/slurm/hpc-slurm.conf   # HPC (CPU) partition
```

Add both to `~/.bashrc` as aliases, or prefix each command:

```bash
SLURM_CONF=/opt1/slurm/gpu-slurm.conf sbatch ...
```

---

## 1. Partitions

| partition | QOS | account | purpose |
|---|---|---|---|
| `gpu_2h` | `gpu` | `gpu` | GPGPU, 1 GPU per job, **2 h wall** |
| `hpc_72h` | `hpc` | — | CPU only, 192 cores/node, 72 h wall |

HPC has a **group cap of 80 CPUs across all users**. If a job sits pending with
`QOSGrpCpuLimit`, reduce `-c` (16 → 8 → 4).

---

## 2. `gpu_2h` is heterogeneous — check the GPU you get

`sinfo` on 2026-11-22:

| nodes | GPU | VRAM | notes |
|---|---|---|---|
| `gpu1 gpu3 gpu24 gpu27 gpu28 gpu29` | RTX 3090 | 24 GB | Ampere, tensor cores |
| `gpu54 gpu55` | Titan RTX | 24 GB | Turing |
| `projgpu33` | RTX 3090 ×3 + **A6000** | 24 / **48 GB** | **mounts `/research/d7`** |
| `gpu7 gpu8` | Titan X | 12 GB | oldest, no tensor cores |
| `gpu9` | Titan X | 12 GB | fine for `/research/d7` |

`sinfo` says `mix` for a node that is actually unusable: `gpu28` returned
`ReqNodeNotAvail` while still listed. Never assume availability from `sinfo` alone.

---

## 3. Storage

| path | size | notes |
|---|---|---|
| `/uac/y22/cyho2` (home) | quota **2442 M**, ~1248 M used | ~1.2 GB free — **not** for datasets or checkpoints |
| `/research/d7/fyp26` | 140 TB (86 TB used) | the project store; **verify the mount per node** |
| `/dev/shm` (per node) | RAM | container + scratch, wiped at job end |

`/research/d7` is **not** mounted uniformly:

- `projgpu33` → **MOUNTED** (verified 2026-11-22)
- `gpu7`, `gpu9` → work
- `gpu8` → autofs hangs
- `gpu36 gpu56 gpu57 gpu59` → cannot resolve it at all

So: pin `-w projgpu33` when you need both a fast GPU and the big filesystem,
or keep the working set small enough for home.

---

## 4. Hard limits (each of these costs a job if you forget)

| flag | constraint | failure you get |
|---|---|---|
| `--mem` | **never pass it** — nodes report `RealMemory=1` | `Requested node configuration is not available` |
| `-c` | ≤ 6 per GPU; 8 needs `--constraint=highcpucount` | same |
| `-t` | ≤ 2 h on `gpu_2h` | job killed at the wall |
| `-w a,b,c` | **one node per job**; multiple hosts → `Node count specification invalid` | submission rejected |
| pending jobs | ~4 per user (observed: 5th submission onward rejected while 4 pending) | silent `FAILED` in a loop |

The 2 h wall is not negotiable — **checkpoint and chain**:

```bash
sbatch gpu.slurm          # writes checkpoint, then:
sbatch --dependency=afterok:$JID gpu.slurm --resume
```

---

## 5. Container

Jobs run through Singularity with a cached PyTorch image:

```
/dev/shm/pytorch.sif   # /dev/shm is per-node, so it may need re-pulling (~5 min)
```

`gpu.slurm` in this repo handles the pull, the dependency install into
`/dev/shm/deps_$JOBID`, and cleanup on exit.

---

## 6. Copy-paste templates

**Check what you actually got** (do this first in every new job):

```bash
sbatch --parsable -p gpu_2h --qos=gpu --account=gpu --gres=gpu:1 -c 2 -t 00:02:00 \
  -w projgpu33 -o ~/probe-%j.out --wrap "
    hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader;
    ls /research/d7/fyp26 >/dev/null 2>&1 && echo MOUNTED || echo NOT_MOUNTED;
    test -f /dev/shm/pytorch.sif && echo IMAGE_PRESENT || echo IMAGE_MISSING"
```

**CPU batch job (MFA, preprocessing, probing sweeps):**

```bash
SLURM_CONF=/opt1/slurm/hpc-slurm.conf sbatch -p hpc_72h --qos=hpc -c 16 \
  -t 4:00:00 -o ~/job-%j.out --wrap "cd ~/final-year-project && ./run.sh"
```

**GPU training job:**

```bash
SLURM_CONF=/opt1/slurm/gpu-slurm.conf sbatch -p gpu_2h --qos=gpu --account=gpu \
  --gres=gpu:1 -c 6 -t 01:45:00 -w projgpu33 gpu.slurm train.py --epochs 10
```

**Monitoring:**

```bash
SLURM_CONF=/opt1/slurm/gpu-slurm.conf squeue -u $USER -o "%.8i %.10T %.6C %R"
SLURM_CONF=/opt1/slurm/gpu-slurm.conf sinfo -p gpu_2h -o "%n %t %G %c"
```

---

## 7. Pre-flight checklist

- [ ] `SLURM_CONF` exported
- [ ] no `--mem`
- [ ] `-c` ≤ 6
- [ ] `-t` ≤ 2 h, with checkpoint/resume in the script
- [ ] one node in `-w`, and it mounts `/research/d7` if the job needs it
- [ ] data + checkpoints in `/research/d7/fyp26`, not home
- [ ] fewer than ~4 jobs already pending under this user
- [ ] script is idempotent — a resumed job must not clobber the previous checkpoint

---

## 8. Practical verdict

Slurm is good for **submitting finished, resumable jobs**; it is poor for
**interactive iteration** — queue waits, a 2 h wall and a pending-job cap make
short debug cycles painful.

- probes / sweeps / MFA → **HPC CPU** partition
- VC training → **`gpu_2h` on `projgpu33`**, checkpoint-chained
- long uninterrupted runs → AutoDL V100S (paid, no wall) as fallback
