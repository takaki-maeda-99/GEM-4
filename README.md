# vla-gemma-4

**English** | [日本語](README.ja.md)

> Wearable Vision-Language-Action assistant on Gemma 4 — a hands-free companion arm prototype for people with limb or visual impairments.

[![Hackathon](https://img.shields.io/badge/Hackathon-Gemma%204%20Good-orange)](https://www.kaggle.com/competitions/gemma-4-good-hackathon/)
[![Backbone](https://img.shields.io/badge/Backbone-Gemma%204%20E2B-blue)](https://www.kaggle.com/models/google/gemma-4)
![Sim benchmark](https://img.shields.io/badge/LIBERO--Spatial-94%25-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

<!-- TODO: docs/images/hero.gif — wearable demo hero -->

## TL;DR

- **What**: A wearable VLA assistant — voice in, action out — built around Gemma 4.
- **Why Gemma 4 + VLA-Adapter**: PLE × VLA-Adapter bridge attention is a unique architectural fit; the LLM stays frozen and only a small adapter / projector / action head is trained → short-window efficient adaptation. An E2B-class open-weight LLM brings on-device offline inference (Jetson) within reach.
- **What works today**: LIBERO-Spatial 94 % in simulation (X-VLA-Adapter v33), plus operator-supervised scripted demos of three real-robot tasks (take from shelf / open lid / hold).
- **How to evaluate in 5 minutes**:
  1. Watch the demo media in [Demo / What it does today](#demo--what-it-does-today)
  2. Read the architecture in [System overview](#system-overview)
  3. Skim [Why Gemma 4 + VLA-Adapter](#why-gemma-4--vla-adapter)
  4. Verify LIBERO 94 % via [Reproducibility](#reproducibility)

## Mission

People with limb or visual impairments need *another arm* — something that can take a cup down from a shelf, open a lid, or hold an object steady, on demand.

We are building exactly that: a single arm worn on the body, with a chest-mounted overview camera and a wrist camera, taking voice instructions in natural language and acting on the physical world while sharing what it sees with the user. The design intent is **semi-autonomous companionship**, not full autonomy — the system extends the user's intent, it does not replace it.

We believe this is the moment to attempt it. Gemma 4's open-weight, small-footprint, on-device performance, combined with VLA-Adapter (Wang et al., 2025) — which keeps the LLM frozen and trains only a small adapter — bring **short-window efficient adaptation** within reach. The ~1.5-month sprint of this hackathon was enough to converge on LIBERO-Spatial 94 %. *This is a research prototype with operator-supervised demos; it is not a medical device.*

## Demo / What it does today

> **Status note**: All real-robot demos shown are **operator-supervised scripted demos**. A physical e-stop and software watchdog are required for any session. GIFs / video links will be filled in as media becomes available.

### Representative tasks (operator-supervised scripted demo)

- **Take from shelf** — <!-- TODO: docs/images/demo_shelf.gif -->
- **Open lid** — <!-- TODO: docs/images/demo_lid.gif -->
- **Hold / support** — sustained assistance, our differentiator vs. typical VLA pick-and-place — <!-- TODO: docs/images/demo_hold.gif -->

### Sim benchmark

- **LIBERO-Spatial 94 %** — X-VLA-Adapter v33, 47 / 50 episodes, 50 episode/suite, 4 tasks × 50 steps
- Training curve: <!-- TODO: docs/images/v33_training_curve.png -->

### Reproducibility

| Artifact | Pointer |
|---|---|
| Train config | [`X-VLA-Adapter/configs/train/libero_spatial_v33.yaml`](./X-VLA-Adapter/configs/train/libero_spatial_v33.yaml) |
| Eval config | [`X-VLA-Adapter/configs/eval/libero_v33_step40000.yaml`](./X-VLA-Adapter/configs/eval/libero_v33_step40000.yaml) |
| Checkpoint | <!-- TODO: HF Hub or Drive direct link --> |
| Eval command | `uv run python scripts/eval.py configs/eval/libero_v33_step40000.yaml` (run from `X-VLA-Adapter/`) |
| Hardware / SW | RTX 6000 Ada / Ubuntu 22.04 / CUDA 12.6 / Python 3.12 / `uv` lockfile committed |
| Normalization stats | Distributed alongside the checkpoint (`norm_stats.json`) |
| Model card | <!-- TODO: link to model card with limitations section --> |

## System overview

> Legend: green solid = shipped / yellow dot-dash = in-progress (stub / smoke / training) / orange dashed = planned / blue = hardware / purple = data store / pink = actor.
> Edges: solid = working path, dotted = path leading into in-progress or planned destinations.

```mermaid
flowchart TB
    User(("User<br/>voice instruction"))

    subgraph HW["Wearable hardware — CAD_Library/"]
        direction LR
        Harness["Body harness"]
        Arm["1-arm reBot B601-DM"]
        Gripper["Custom gripper"]
        ChestCam["Chest overview cam"]
        WristCam["Wrist cam"]
        Harness --- Arm
        Harness --- ChestCam
        Arm --- Gripper
        Arm --- WristCam
    end

    subgraph DC["MimicRec — collection / replay / inference client"]
        direction TB
        MR_Adapter["Robot adapters<br/>SO-101 / reBotArm / Isaac Sim"]
        MR_Teleop["Teleop /<br/>hand-teach"]
        MR_Replay["Replay +<br/>safety watchdog"]
        MR_StubAnno["In-app annotator<br/>(stub)"]
        MR_HumanAdapter["Human-video adapter<br/>skeleton → EE Δ"]
        MR_Out[("LeRobot v3<br/>episodes")]
        MR_Adapter --> MR_Teleop --> MR_Out
        MR_StubAnno -.-> MR_Out
        MR_HumanAdapter -.-> MR_Out
    end

    subgraph ANNO["MimicAnno — offline subtask annotation"]
        direction TB
        MA_P1["Phase 1<br/>signal boundaries"]
        MA_P2["Phase 2<br/>Gemma 4 VLM labeling"]
        MA_P3["Phase 3<br/>SAM3 tracking"]
        MA_P4["Phase 4<br/>Viterbi smoothing"]
        MA_P5["Phase 5<br/>autonomous label + edit UI"]
        MA_Out[("SARM-trainable<br/>+ subtask_index")]
        MA_P1 --> MA_P2 --> MA_P3 --> MA_P4 --> MA_Out
        MA_P5 -.-> MA_Out
    end

    subgraph TRAIN["X-VLA-Adapter — training"]
        direction TB
        T_Vision["SigLIP vision<br/>(chest + wrist)"]
        T_Gemma["Gemma 4 E2B + PLE<br/>frozen"]
        T_Adapter["VLA-Adapter<br/>bridge attention<br/>trainable"]
        T_Head["L1 action head<br/>trainable"]
        T_Ckpt[("checkpoint +<br/>norm stats")]
        T_Vision --> T_Adapter
        T_Gemma <--> T_Adapter
        T_Adapter --> T_Head --> T_Ckpt
    end

    subgraph INFER["Inference — X-VLA-Adapter Phase 0 server"]
        direction LR
        I_API["FastAPI /predict<br/>HoldPosition stub OK<br/>real predictor pending"]
        I_Quant["NF4 / HQQ<br/>quantization (planned)"]
        I_Jetson["Jetson on-device<br/>real-time loop"]
        I_API --> I_Quant -.-> I_Jetson
    end

    subgraph FUTURE["Architectural roadmap"]
        direction TB
        R_Hier["Hierarchical inference<br/>subtask planner + executor"]
        R_Cross["Cross-embodiment<br/>full X-VLA<br/>(currently training)"]
    end

    User -- voice --> I_API
    User -- voice --> MR_Teleop
    ChestCam --> MR_Adapter
    WristCam --> MR_Adapter
    Arm <--> MR_Adapter

    MR_Out --> MA_P1
    MA_Out --> T_Vision
    MA_Out --> T_Gemma
    MR_Out --> T_Vision

    T_Ckpt --> I_API
    I_API -- action chunk --> MR_Replay
    MR_Replay -- joint cmd --> Arm

    MA_Out -.-> R_Hier
    R_Hier -.-> I_API
    T_Ckpt -.-> R_Cross

    classDef shipped fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef inprogress fill:#fff8e1,stroke:#f9a825,color:#f57f17,stroke-dasharray:5 2 2 2,stroke-width:2px
    classDef planned fill:#fff3e0,stroke:#ef6c00,color:#e65100,stroke-dasharray:5 5
    classDef hardware fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef data fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef actor fill:#fce4ec,stroke:#ad1457,color:#880e4f

    class User actor
    class Harness,Arm,Gripper,ChestCam,WristCam hardware
    class MR_Adapter,MR_Teleop,MR_Replay,MA_P1,MA_P2,MA_P3,MA_P4,T_Vision,T_Gemma,T_Adapter,T_Head shipped
    class I_API,R_Cross,MA_P5,MR_StubAnno inprogress
    class MR_HumanAdapter,I_Quant,I_Jetson,R_Hier planned
    class MR_Out,MA_Out,T_Ckpt data
```

Primary data flows: (1) user voice → MimicRec inference client → X-VLA-Adapter Phase 0 server, (2) chest + wrist cameras → SigLIP → VLA-Adapter bridge attention with frozen Gemma 4 → action head → 1-arm replay, (3) data loop: episodes collected by MimicRec → annotated by MimicAnno → fed back to X-VLA-Adapter for re-training. On-device offline operation (Jetson class) is the **target**; training currently runs on cloud or on-prem GPU.

<!-- Optional: docs/images/system_architecture.png — a more polished hand-drawn architecture diagram, when available -->

## Why Gemma 4 + VLA-Adapter

Where Gemma 4 runs in this project:

| Where | What |
|---|---|
| `X-VLA-Adapter` | The VLA backbone (E2B) at inference time; vision / action tokens are injected into each layer via bridge attention |
| `MimicAnno` Phase 2 | Image-text-to-text VLM that labels the subtask phase of each segment offline |
| Multilingual instructions | Natural language instructions in both Japanese and English are passed in directly |

Five reasons we chose this stack:

1. **Designed for offline on-device** — At E2B, Gemma 4 is in reach of Jetson-class hardware. Training-time memory is compressed via `bitsandbytes` 8-bit AdamW (`X-VLA-Adapter/src/vla_project/training/optim.py`); inference-side NF4 / HQQ quantization and the Jetson real-time loop both remain on the **roadmap**.
2. **PLE × VLA-Adapter bridge attention** — Gemma 4's Per-Layer Embeddings sit at every layer; VLA-Adapter injects vision and action tokens into every layer through bridge attention. The two structures stack naturally on top of each other. Our PAD-placeholder-ID workaround for routing custom tokens through PLE is a concrete engineering contribution.
3. **Short-window efficient adaptation via VLA-Adapter** — VLA-Adapter (Wang et al., 2025) freezes the LLM and trains only a small adapter + projector + action head. For our timeline this means:
   - fine-tuning converges on hundreds-to-thousands of episodes, orders of magnitude cheaper than full fine-tune
   - transfer to new robot embodiments and new tasks is a matter of swapping adapters
   - combined with LoRA (r = 16 / 64), we ran a v3 → v37 architecture sweep within the ~1.5-month hackathon window
   - within that window (~46 days, ~10 days remaining at time of writing), v33 reaching LIBERO 94 % depended directly on this efficiency
4. **Broad world knowledge + multilingual** — object names, physical intuition, and bilingual (ja / en) instruction handling all serve as a generalization scaffold.
5. **Open weights** — LoRA, quantization, and architecture surgery are unrestricted. Our v25 → v37 architecture sweep (`X-VLA-Adapter/configs/train/`) depends on this.

## The stack — 4 components

### `X-VLA-Adapter/` — VLA model & training & inference

**Role**: SigLIP + Gemma4-E2B + per-domain projector + L1 action head — the VLA policy itself. LIBERO benchmark is the primary axis; the same repo also handles real-robot deployment.

**Capabilities**:
- Switch every architecture revision from a single train YAML (v3 → v37: LoRA / soft-prompt-in-LLM / wrist-into-LLM / DA-2-MLP / etc.)
- Auto-sweep 50 ep × 4 suites × N steps from a single eval YAML
- Phase 0 FastAPI inference server (compatible with the MimicRec `/predict` contract; HoldPosition stub today, real model predictor pending v36)
- 4-domain RLDS + LeRobot dataloader, shared Q99 statistics
- Layer-wise freezing, per-param-group LR coefficients, checkpoint resume
- (planned) NF4 / HQQ quantization, Jetson on-device real-time loop

**Design principles** (see `CLAUDE.md` in this submodule): strict separation between `data` / `models` / `policies` / `training` / `evaluation` / `deployment` / `robots`, fully config-driven, robust to overnight sweeps.

**Headline result**: LIBERO-Spatial v33 = 94 % (47 / 50).

**Internal flow**:

```mermaid
flowchart LR
    D[("LeRobot v3 / RLDS<br/>+ subtask_index")]
    I["instruction text"]
    V["SigLIP<br/>chest + wrist"]
    L["Gemma 4 E2B + PLE<br/>frozen"]
    A["VLA-Adapter<br/>bridge attention<br/>trainable"]
    H["L1 action head<br/>trainable"]
    O["EE Δ + gripper<br/>action chunk"]
    D --> V
    I --> L
    V --> A
    L <--> A
    A --> H --> O
    classDef frozen fill:#eceff1,stroke:#455a64,color:#263238
    classDef trainable fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef data fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    class L frozen
    class V,A,H trainable
    class D data
```

**Details**: → [`X-VLA-Adapter/README.md`](./X-VLA-Adapter/README.md)

### `MimicRec/` — local-first data collection web app

**Role**: A web app that collects IL data from a physical robot and outputs LeRobot v3 datasets. Teleop / hand-teach / replay / VLA inference flow through one stack.

**Capabilities**:
- Teleop (leader arm / keyboard / sim) recording; hand-teach with gravity compensation + gripper friction compensation (reBotArm)
- Replay with synchronized arm + gripper playback and a safety watchdog (joint position jump / velocity / acceleration triple-gate)
- Drop in any VLA model on the live robot via a single VLA HTTP contract YAML
- LeRobot v3 zip download, episode review (success / failure label)
- Settings UI: device discovery, calibration status, adapter config editing
- ~250 backend tests, 500 Hz motor control isolated in a separate daemon

**Supported hardware**: SO-101 / reBot Arm B601-DM / Mock / Isaac Sim (Franka verified). New robots only need a single `RobotAdapter` protocol implementation to appear in the UI.

**Internal flow**:

```mermaid
flowchart LR
    Op["Operator<br/>(leader / kbd / sim)"]
    HW["Robot HW<br/>SO-101 / reBotArm / Sim"]
    subgraph Backend["MimicRec backend"]
        Adapter["RobotAdapter"]
        SM["SessionManager<br/>(asyncio control loop)"]
        Replay["Replay +<br/>safety watchdog"]
        VLA["VLA HTTP client"]
        Writer["LeRobot v3 writer"]
        StubAnno["In-app annotator<br/>(stub)"]
    end
    Out[("LeRobot v3<br/>episodes")]
    Predict["External VLA server<br/>/predict"]

    Op --> SM
    HW --> Adapter --> SM
    SM --> Writer --> Out
    Out --> Replay --> HW
    SM <--> VLA
    VLA <--> Predict
    StubAnno -.-> Out

    classDef ext fill:#fff3e0,stroke:#ef6c00,color:#e65100
    classDef data fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    classDef hw fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef inprogress fill:#fff8e1,stroke:#f9a825,color:#f57f17,stroke-dasharray:5 2 2 2,stroke-width:2px
    class HW hw
    class Out data
    class Predict ext
    class StubAnno inprogress
```

**Details**: → [`MimicRec/README.md`](./MimicRec/README.md)

### `MimicAnno/` — offline subtask annotation

**Role**: An offline tool that adds subtask boundaries + labels to a LeRobot v3 episode and exports a SARM-trainable dataset.

**Capabilities**:
- **Phase 1** — signal-driven boundary detection (gripper transitions / EEF velocity / action-norm change points)
- **Phase 2** — Gemma 4 VLM labels each segment's phase (allowed-label set + JSON schema enforced; fail-fast on invalid output)
- **Phase 3** — SAM3 object tracking driven by task text, fused into the boundary score
- **Phase 4** — same-label merge + minimum-duration absorb + Viterbi relabel
- **Export** — per-frame `subtask_index` + episode-level subtask list, atomic publish, idempotent re-run
- React / Vite read-only timeline + waveform viewer
- YAML adapter for any LeRobot v3 layout (so100 / koch / aloha / SO-101 generic)
- (in-progress) Phase 5 — autonomous labeling + edit UI

**Internal flow**:

```mermaid
flowchart LR
    In[("LeRobot v3 episode<br/>+ task text")]
    P1["Phase 1<br/>signal boundaries<br/>(gripper / EEF / action)"]
    P2["Phase 2<br/>Gemma 4 VLM<br/>phase labeling"]
    P3["Phase 3<br/>SAM3 object tracking"]
    P4["Phase 4<br/>same-label merge<br/>+ Viterbi"]
    P5["Phase 5<br/>autonomous label<br/>+ edit UI"]
    Pub["atomic publish<br/>(idempotent)"]
    Out[("SARM-trainable LeRobot v3<br/>+ subtask_index<br/>+ sidecar parquet")]

    In --> P1 --> P2 --> P3 --> P4 --> Pub --> Out
    P5 -.-> Pub

    classDef vlm fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef inprogress fill:#fff8e1,stroke:#f9a825,color:#f57f17,stroke-dasharray:5 2 2 2,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
    class P2,P3 vlm
    class P5 inprogress
    class In,Out data
```

**Details**: → [`MimicAnno/README.md`](./MimicAnno/README.md)

### `CAD_Library/` — wearable hardware

**Role**: All physical components for the wearable, as open CAD. **Not a submodule** — it lives as a regular directory inside this repo, with `SLDPRT` / `SLDASM` / `STEP` files tracked through git-lfs.

**Contents**:
- `Robot_Arm/reBot_B601_DM_v1.0_20260331.step` — the arm itself (STEP)
- `Gripper/gripper.SLDASM` — custom lightweight gripper
- `Harness/bodey harness.SLDASM` — body-mounted harness
- `Data_Collection_Device/{ver1,ver2}/` — chest camera mount + data-collection rigs

**Viewing**: SolidWorks opens everything natively; without SolidWorks, the STEP file can be inspected with FreeCAD or any standards-compliant CAD viewer.

## Quickstart

```bash
git clone --recurse-submodules <url> && cd vla-gemma-4
git submodule update --init --recursive   # if you forgot --recurse-submodules
git lfs install && git lfs pull            # fetch CAD_Library STEP / SLDASM
```

Three entry points:

| What you want to do | Where to go |
|---|---|
| Train (LIBERO sim) | → [`X-VLA-Adapter/README.md#Training`](./X-VLA-Adapter/README.md#training) |
| Collect data (real / mock / sim) | → [`MimicRec/README.md#Quick-start`](./MimicRec/README.md#quick-start) |
| Annotate (existing dataset) | → [`MimicAnno/README.md#Quickstart`](./MimicAnno/README.md#quickstart) |

**What runs from root**: nothing today. Each submodule README is the single source of truth for its own runtime. An end-to-end (collect → annotate → train → infer → replay) command sequence will be added later.

**Prerequisites**: Ubuntu 22.04 / 24.04, Python 3.12 (`uv` will pull it), Node 20+ (MimicRec frontend), NVIDIA GPU with CUDA 12.6+ (X-VLA-Adapter training). macOS / WSL may work for some submodules but is unverified.

## Safety & Scope

### Current safety measures

- **Physical emergency stop (E-stop)**: <!-- TODO: button position / response latency / stopping torque, to be filled in after on-robot measurement -->
- **Software watchdog**: replay path enforces a triple gate on joint position jump / velocity / acceleration (`MimicRec/configs/robot/<robot>.yaml` `replay:` block); the daemon adds further `safety:` clamps in `configs/rebotarm_daemon.yaml`
- **Soft stop (graceful halt)**: <!-- TODO: define behavior triggered by spoken "stop" / "止まって" once implemented -->
- **Action rate limit**: <!-- TODO: maximum EE Δ per step and gripper velocity, to be measured on-robot -->
- **Operator presence**: every real-robot demo session has an operator within reach of the E-stop

### Scope (what the project is, and is not)

- This is a **research prototype**, NOT a medical device or certified assistive device.
- All real-robot demos shown are **operator-supervised**; the safety measures above are required for any session.
- Unsupervised home deployment is **explicitly out of scope**.
- Camera and voice data: by default everything stays on-device; the runtime path performs no cloud upload.
- Intended assistance is single-task, short-horizon manipulation while the operator is conscious and able to halt the system.
- We do not claim accessibility certification, regulatory approval, or clinical efficacy.

## Status & Roadmap

Three buckets, matching the system overview mermaid: **Shipped / In progress / Roadmap**.

### Shipped

- **(shipped)** LIBERO-Spatial 94 % (X-VLA-Adapter v33) — see [Reproducibility](#reproducibility) for config + eval cmd + checkpoint
- **(shipped)** Operator-supervised scripted demo of three real-robot tasks (take from shelf / open lid / hold)
- **(shipped)** MimicAnno Phase 1-4 annotation pipeline (signal boundary / Gemma 4 VLM / SAM3 / Viterbi)
- **(shipped)** MimicRec end-to-end (collect → review → replay) on SO-101 / reBot Arm / Isaac Sim
- **(shipped)** Wearable hardware prototype (`CAD_Library/`) — STEP / SLDASM / SLDPRT published
- **(shipped)** 8-bit AdamW (`bitsandbytes`) for training-time optimizer-state compression

### In progress

- **(in-progress)** X-VLA-Adapter Phase 0 inference server — HoldPosition stub passes wire smoke; the real-model `XVLAAdapterChunkPredictor` is a stub awaiting v36 checkpoints
- **(in-progress)** Cross-embodiment X-VLA — v34 / v35 multi-domain RLDS **currently training**, eval pending
- **(in-progress)** MimicAnno Phase 5 — autonomous labeling + edit UI (autonomy mode in flight)
- **(in-progress)** MimicRec in-app annotator — stub (real implementation lives in MimicAnno)

### Roadmap — NOT IMPLEMENTED

> The items below are **not started**. They are listed here so they cannot be confused with shipped capability.

- **Scale data**:
  - **(planned)** New MimicRec adapter: first-person human hand video → hand-skeleton estimation → EE Δ + gripper representation → action data → transfer learning to scale data collection
- **Architecture**:
  - **(planned)** Hierarchical inference using MimicAnno subtasks: high-level subtask planner + low-level skill executor (a stage past the current monolithic VLA). MimicAnno already produces the training data; the inference-side hierarchy is unbuilt.
- **Deployment**:
  - **(planned)** Inference-side NF4 / HQQ quantization. Smoke scripts from the pre-refactor monolithic layout (e.g. `test_nf4_26b_smoke.py`) were removed in commit `084d0ba` and have not been ported to X-VLA-Adapter yet.
  - **(planned)** Jetson on-device real-time inference loop (TensorRT + quantized inference integration).

## Acknowledgements

- **Hackathon**: [The Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/) — Kaggle × Google DeepMind, 2026-04-02 → 2026-05-18
- **Upstream**: VLA-Adapter (Wang et al., 2025) / X-VLA / SigLIP / Gemma 4 / LeRobot / SAM3 / LIBERO

## License

- Root repository: **Apache-2.0** (see [`LICENSE`](./LICENSE))
- All submodules (`MimicRec`, `MimicAnno`, `X-VLA-Adapter`): **Apache-2.0**
- `CAD_Library/` SLDPRT / SLDASM / STEP: governed by the root LICENSE
- Upstream libraries (LeRobot, SAM3, etc.) retain their own licenses
