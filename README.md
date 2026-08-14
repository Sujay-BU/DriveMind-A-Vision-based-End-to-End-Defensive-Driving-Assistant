# DriveMind: a vision-based end-to-end autonomous defensive driver

Implementation of my [thesis](https://open.bu.edu/items/be7595bb-8bc1-4827-beb8-c531778cfd73) titled "DriveMind: a vision-based end-to-end autonomous defensive driver".  
  
My work builds off of [Jia et al. (2025)](https://arxiv.org/abs/2503.07656), making 5 modifications to the architecture to make it safety-aware through the novel benchmark for defensive driving proposed in my thesis.


---

## What is here

| Piece | Path |
|---|---|
| Table 1 thresholds and per-frame compliance scoring | `dav/metrics/` |
| Rule-based defensive expert (Table 1 as hard constraints) | `dav/data/expert.py` |
| CARLA collector → nuScenes-shaped dataset | `dav/data/collector.py` |
| Safety-aware DriveTransformer (all 5 modifications) | `dav/models/` |
| Safety Huber loss + full multi-task criterion | `dav/losses/` |
| Training loop, tiny + large configs | `dav/train.py`, `configs/` |
| Bench2Drive agent and evaluation harness | `dav/agents/`, `dav/eval_b2d.py` |
| Training dashboard | `gui/` |
| Best-episode video export | `scripts/export_best_video.py` |
| Crash-tolerant collection wrapper | `scripts/collect_supervised.sh` |
| Tests | `tests/test_dav.py` |

---

## Setup

```bash
conda env create -f environment.yml
conda activate dav
pytest -q                      # 40 tests, ~5 s
```

Python is pinned to 3.10 because that is the newest interpreter with a
published `carla==0.9.15` wheel, and it is new enough for current PyTorch.

CARLA is expected at:

```bash
export CARLA_ROOT=~/Desktop/github_projects/3D_reconstruction/vendor/CARLA_0.9.15
```


---

## The  pipeline

### 1. Collect

```bash
# Pilot: 8 episodes x 1000 frames, a few hours, ~6-9 GB.
bash scripts/collect_supervised.sh configs/collect_pilot.yaml

# Full, exactly as Chapter 3 specifies: 50 x 3000 across 11 weathers x 8 towns.
# 2.5-4 days and 300-600 GB. Launch it detached.
nohup bash scripts/collect_supervised.sh configs/collect_full.yaml > collect.log 2>&1 &
```

**Use the supervised script for anything longer than one town.** A map load is
minutes of heavy I/O and can crash the simulator; the supervisor restarts CARLA
and the collector resumes from the episodes already on disk rather than
starting over (D51). `scripts/collect_data.sh` still exists for a single
short run where that is not worth the wrapper.

The script starts CARLA if it is not already running.

Episodes are accepted on a **compliance rate**, at least 70% of frames free of
every Table 1 infraction (`min_compliance`), and must also cover ground
(`min_mean_speed`), because compliance alone is trivially maximised by not
moving. Chapter 3's literal "a single infraction is a failure" rule accepts
nothing against a live simulator; see D22. Failing episodes are re-run with a
new seed, and `data/<root>/manifest.json` records what was accepted, with
per-metric compliance for each episode.

Each episode also writes `trace_<episode>.csv`: per-frame ego state, control,
and violations. That file is how nearly every expert defect in D23–D67 was
found, start there if the expert misbehaves. Aggregate compliance tells you
*that* something is wrong; the trace tells you which frame and why.

### 2. Train

```bash
bash scripts/train.sh configs/dav_tiny.yaml      # one 6 GB GPU
bash scripts/train.sh configs/dav_large.yaml     # cluster; see limitation L3
```

Ablate a single modification by flipping one flag:

```bash
python -m dav.train --config configs/dav_tiny.yaml \
  --run-name no_temporal --overrides model.use_safety_temporal=false
```

### 3. Watch

```bash
bash scripts/run_gui.sh 8080
# over SSH:  ssh -L 8080:127.0.0.1:8080 <host>
```

Live loss curves, per-metric defensive-driving compliance, safety-query
diagnostics, dataset audit, run comparison, Bench2Drive panel, and playback of
exported videos. Light and dark themes; "Table view" shows the numbers behind
every chart.

### 4. Evaluate on Bench2Drive

```bash
git clone https://github.com/Thinklab-SJTU/Bench2Drive.git ../Bench2Drive

python -m dav.eval_b2d run \
  --bench2drive ../Bench2Drive --carla "$CARLA_ROOT" \
  --checkpoint runs/<run>/best.pt --out-dir runs/<run>
```

Writes `runs/<run>/bench2drive.json`, which the dashboard picks up
automatically and scores against DS 72.46 / SR 44.01. Budget 20–40 hours on a
consumer GPU.

### 5. Export the best episode as one video

```bash
python scripts/export_best_video.py \
  --run runs/<run> --data data/dav_pilot --out-dir outputs/videos
```

Writes a single MP4 into a separate directory, plus a JSON sidecar. Each frame
carries the six-camera mosaic, the planned path against the expert's, live
Table 1 metric bars with the threshold marked, and a compliance badge. Rank
episodes with `--rank-by compliance|ade|driving`.

Two flags matter more than they look:

- **`--max-frames` defaults to 900**, i.e. 45 seconds. A five-minute video needs
  `--max-frames 6000`; without it a long episode is silently truncated.
- **The exporter refuses to overwrite an existing file.** Pass `--name <stem>`
  to write alongside it, or `--force` to replace it. Re-exporting the same run
  and episode, a longer take, a different checkpoint, would otherwise
  silently destroy a recording that took an hour to produce.

### Traffic that the ego actually meets

Random spawn points scatter vehicles across the map, so whether the ego ever
encounters a lead vehicle is left to chance: one pilot episode registered one
in 2.2% of frames with forty cars present. `lead_vehicles` and
`oncoming_vehicles` seed traffic directly ahead in the ego's lane and in the
opposing carriageway, which takes that to 30%+ and actually exercises the
following-distance and give-way rules. See `configs/collect_traffic_town10.yaml`.

Density has to be balanced against gridlock: 70 scattered vehicles plus 12
seeded deadlocks downtown Town10HD over five minutes (D63).

### Which collection config to use

| config | what it is for |
|---|---|
| `collect_smoke.yaml` | one short episode, `allow_infractions: true`, diagnosing the expert. Prints every metric's violation rate instead of aborting on the first one |
| `collect_pilot.yaml` | the 8-town pilot as Chapter 3 intends. Needs more than 6 GB VRAM (D54) |
| `collect_pilot_town01.yaml` | the pilot this machine can run: 8 weathers, one town. **This produced the dataset** |
| `collect_full.yaml` | 50 × 3000 across 11 weathers × 8 towns. Days, hundreds of GB |
| `collect_traffic_town10.yaml` | 45 s dense-traffic run on a second map, seeded lead and oncoming vehicles |
| `collect_traffic_5min.yaml` | 5 min at the *higher* density. Retained as a negative result: fails the 70% gate repeatedly |
| `collect_traffic_5min_record.yaml` | 5 min, gate disabled, for recording only. **Not training data** |
| `collect_5min_gated.yaml` | 5 min with the gate enforced. **Produced the accepted 87.83% run** |

---

## Status

[![Result Video](https://raw.githubusercontent.com/Sujay-BU/DriveMind-A-Vision-based-End-to-End-Defensive-Driving-Assistant/main/outputs/videos/Preview.jpg)](https://raw.githubusercontent.com/Sujay-BU/DriveMind-A-Vision-based-End-to-End-Defensive-Driving-Assistant/main/outputs/videos/dav_5min_Town10HD_redlight_fixed.mp4)

Verified environment, model forward/backward,
dashboard, video export, 40 tests, **data collection against a live CARLA
server**, and **training on the real collected data**.

A **pilot dataset has been collected**: 8 episodes x 1000 frames on Town01
across 8 weather presets, 48000 images, 1.6 GB. Every episode satisfies the
70% compliance rule, mean **78.1%**, minimum **73.1%**. Training on it for
three epochs reaches a validation ADE of **1.117 m**.

A **five-minute continuous run on Town10HD** with dense traffic (vehicles
seeded ahead of the ego and in the opposing lane) scores **87.83% with the gate
enforced**, and zero red-light violations.


**Recorded videos** in `outputs/videos/`:

| file | what it is |
|---|---|
| `dav_5min_Town10HD_redlight_fixed.mp4` | 4.95 min, 87.83% compliance, gate enforced, no red-light violations, **the one to watch** |

Not yet run: training on real data at scale (Over 10M images).
