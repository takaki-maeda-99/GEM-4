# GEM-4

**English** | [日本語](README.ja.md)

> A Gemma 4-based wearable Vision-Language-Action assistant: a hands-free companion arm prototype that sees, understands voice instructions, and moves to help people with limb or visual impairments.

[![Backbone](https://img.shields.io/badge/Backbone-Gemma%204%20E2B-blue)](https://www.kaggle.com/models/google/gemma-4)
[![Pretrain](https://img.shields.io/badge/Pretrain-OXE%20%2B%20LIBERO%204--suite-blue)](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)
![LIBERO 4-suite avg](https://img.shields.io/badge/LIBERO%204--suite%20avg-74%25-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

![alt text](media/HEROv1.png)

## Overview

A cup is just out of reach. A lid needs two hands. An object needs to be held steady for a few seconds. Small moments like these appear again and again in daily life. This project asks what it would take to have "another arm" present at exactly those moments.

We explore that idea as a wearable one-arm robot with a chest camera, a wrist camera, and natural-language voice instructions. The user says "take it," "open it," or "hold this." The system reads the intent, looks at the scene, and turns the request into a short physical action. The goal is **semi-autonomous assistance**: not a fully autonomous robot, but something closer to an extension of the user's own body and intent.

The technical bet is **SigLIP + Gemma 4-E2B + per-domain projectors + L1 action head**, connected through an action-generation adapter that cross-attends over each Gemma 4 layer. The Gemma 4 backbone stays frozen and only the smaller modules around it are trained, so we can iterate quickly and recover from failed experiments. A single base model is pretrained on **Open-X-Embodiment + LIBERO at ~8 : 2** and then fine-tuned per task / suite, reaching **LIBERO 4-suite average 74 %** in simulation (spatial 72 %, object 92 %, goal 89 %, long 43 % — 10 episodes per task × 10 tasks). The inference stack also runs on Jetson, so on-device use is possible without depending on the cloud.

This is a research prototype with operator-supervised demos. It is **not** a medical device or certified assistive device.

## What Is VLA?

**Vision-Language-Action (VLA)** is a robot policy that connects three things:

- **Vision**: what the cameras see.
- **Language**: what the user asks for, such as "open the lid" or "hold this."
- **Action**: the robot motion to execute next.

For this project, VLA is the bridge from a user's instruction and camera images to short robot action chunks. It is not a fully autonomous household robot; it is a learned controller inside a supervised assistive system.

## Capabilities

- **A VLA policy can be trained and evaluated end-to-end**: a single base ([`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)) is pretrained on **OXE + LIBERO (~8 : 2)** and then fine-tuned per LIBERO suite. With Gemma 4 frozen, the small projector / action-head modules are what get trained, and on FT step 50 k the **LIBERO 4-suite average is 74 %** (spatial 72 %, object 92 %, goal 89 %, long 43 % — 100 episodes per suite). Object and Goal in particular show that the policy can convert Gemma 4's language / vision understanding into successful actions.
- **The VLA stack is built to expand across domains**: GEM-4-VLA is designed around multi-domain RLDS / LeRobot data with per-domain input / output projectors, not a single benchmark-only path. Cross-embodiment / multi-domain transfer is part of the supported scope, making this the foundation for scaling to more robots and tasks.
- **Voice in, robot action out, on-device**: the `raspi_for_vla` client picks up a "hey GEM"-triggered voice instruction, ships the audio to `whisper_hackathon` on Jetson for transcription, and the resulting text is fed to the VLA `/predict` server alongside the chest / wrist camera frames. Cameras, mic audio, transcription, and policy inference can all stay on-device — no cloud round-trip required.
- **MimicRec makes VLA data collection practical**: teleop, hand-teach, replay, review, LeRobot v3 export, and VLA `/predict` integration are brought into one local-first web app. By abstracting the robot interface, the same flow runs across real reBot Arm, SO-101, Isaac Sim, and mock setups — just add a per-robot control adapter. Demo site: <https://takaki-maeda-99.github.io/MimicRec/>.
- **MimicAnno turns raw episodes into richer training data**: subtask boundaries are detected from gripper open / close events plus EEF velocity / acceleration / action-norm changes, then each segment is tracked with SAM3 and labeled by an Unsloth-QLoRA-tuned Gemma 4 (verb / object / target / confidence). A second pipeline reconstructs EEF position, orientation, and pinch distance from a first-person GoPro video using MediaPipe hand landmarks + UniDAC metric depth — a path toward training data generated from human demonstrations alone.

<!-- TODO: add demo GIFs / videos under media/ -->


## Reproducibility

| Artifact | Pointer |
|---|---|
| Pretrain base | [`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE) — OXE 9 datasets + LIBERO 4-suite mix, `step_100000` |
| FT checkpoints | [`GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial), [`-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object), [`-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal), [`-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10) |
| Train config (spatial example) | [`GEM-4-VLA/configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml`](./GEM-4-VLA/configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml) |
| Eval config (spatial step 50k) | [`GEM-4-VLA/configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml`](./GEM-4-VLA/configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml) |
| Eval command | `uv run python scripts/eval.py configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml` from `GEM-4-VLA/` |
| Eval protocol | 10 episodes / task × 10 tasks = **100 episodes per suite**, headless MuJoCo |
| FT recipe | `bs=8 × 2 GPU × accum=2 = eff bs 32` (spatial / object / goal); `bs=8 × 4 GPU × accum=4 = eff bs 128` (libero_10) |
| Hardware / SW | RTX 6000 Ada / Ubuntu 22.04 / CUDA 12.6 / Python 3.12 / `uv` lockfile |
| Normalization stats | Distributed alongside each checkpoint as `norm_stats.json` |

## System
<img width="1017" height="712" alt="image" src="https://github.com/user-attachments/assets/6ef4c7d2-b4ad-46ab-9ab7-1a137df98cf5" />

## Repository Layout

| Path | Role | Details |
|---|---|---|
| [`GEM-4-VLA/`](./GEM-4-VLA/README.md) | Robot policy model, training, evaluation, and `POST /predict` inference server. Headline result: LIBERO 4-suite avg 74 % (FT step 50 k). | [`README`](./GEM-4-VLA/README.md) |
| [`MimicRec/`](./MimicRec/README.md) | Local-first web app for teleop, hand-teach, replay, review, and LeRobot v3 dataset export. | [`README`](./MimicRec/README.md) |
| [`MimicAnno/`](./MimicAnno/README.md) | Offline pipeline for subtask boundary detection, Gemma 4 VLM labeling, SAM3 tracking, Viterbi smoothing, and export. | [`README`](./MimicAnno/README.md) |
| [`raspi_for_vla/`](./raspi_for_vla/README.md) | Raspberry Pi 5 client: USB camera / mic capture, GPIO or wake-word triggered recording, sends observations to Jetson and receives transcripts. | [`README`](./raspi_for_vla/README.md) |
| [`whisper_hackathon/`](./whisper_hackathon/README.md) | Jetson AGX Orin Whisper transcription pipeline (`faster-whisper`); receives audio from the Raspi and returns text to drive the VLA instruction. | [`README`](./whisper_hackathon/README.md) |
| `CAD_Library/` | Wearable hardware CAD: arm, gripper, harness, and camera / data-collection fixtures. | SolidWorks / STEP files via git-lfs |

## Quickstart

```bash
git clone --recurse-submodules <url> && cd vla-gemma-4
git submodule update --init --recursive   # if --recurse-submodules was omitted
git lfs install && git lfs pull            # fetch CAD_Library STEP / SLDASM files
```

Then choose the relevant entry point:

| Goal | Start here |
|---|---|
| Train / evaluate on LIBERO | [`GEM-4-VLA/README.md`](./GEM-4-VLA/README.md) |
| Collect or replay robot data | [`MimicRec/README.md`](./MimicRec/README.md) |
| Annotate an existing dataset | [`MimicAnno/README.md`](./MimicAnno/README.md) |

There is no root-level end-to-end command yet. Each submodule README is the source of truth for its own runtime.

**Prerequisites**: Ubuntu 22.04 / 24.04, Python 3.12, Node 20+ for the MimicRec frontend, NVIDIA GPU + CUDA 12.6+ for GEM-4-VLA training. macOS / WSL may work for some components but is unverified.

## Safety & Scope

- This is a **research prototype**, not a medical device or certified assistive device.
- Real-robot demos are **operator-supervised** and require a physical E-stop plus software watchdog.
- Unsupervised home deployment is explicitly out of scope.
- Runtime camera and voice data stay on-device by default; Jetson offline inference performs no cloud upload.
- The intended assistance is short-horizon, single-task manipulation while a human can stop the system.
- We do not claim accessibility certification, regulatory approval, or clinical efficacy.

## Status & Roadmap

**Shipped**

- GEM-4-VLA v47: LIBERO 4-suite avg **74 %** at FT step 50 k (spatial 72 %, object 92 %, goal 89 %, long 43 %), single OXE+LIBERO pretrain base shared across suites.
- Operator-supervised scripted demos for take from shelf, open lid, and hold / support on the reBot Arm wearable.
- MimicRec collect / review / replay flow on SO-101, reBot Arm, and Isaac Sim, plus a public demo site.
- MimicAnno subtask annotation (gripper / EEF signal + SAM3 + QLoRA-tuned Gemma 4) and human-video EEF estimation (MediaPipe + UniDAC).
- Voice pipeline: `raspi_for_vla` (Pi 5, "hey GEM" wake-word + GPIO) ↔ `whisper_hackathon` (Jetson `faster-whisper`) ↔ VLA `/predict`.
- VLA `/predict` contract with both `xvla_adapter` (real ckpt) and `hold_position` (no-GPU smoke) predictors.
- Jetson on-device offline inference.
- Wearable CAD prototype in `CAD_Library/`.

**Covered in this repo**

- Real-checkpoint inference is handled in the GEM-4-VLA `serve.py` path; FT checkpoints are hosted on Hugging Face under `takaki99/GEM-4-FT-*`.
- Cross-embodiment / multi-domain training is handled in the GEM-4-VLA per-domain projector scope (e.g. ReBotArm single-task FTs over the OXE pretrain base).
- Autonomous labeling and edit UI sit inside the MimicAnno expansion scope.

**Roadmap, not yet implemented**

In this project we integrated the core pieces of a wearable VLA system — data collection infrastructure, annotation tools, and the real-robot interface — but several aspects have not been fully validated and several areas still need real work. The directions below come from the [`KaggleArticle.md`](./KaggleArticle.md) "What Remains to Be Done" discussion.

*Model side*

- **Validate cross-embodiment at scale**: cross-embodiment learning is supported in the codebase, but we have not yet evaluated, at scale, how much shared knowledge and representation actually transfers across multiple robot embodiments.
- **Lift Long-horizon performance**: LIBERO `long` is still our weakest suite (43 %). The priority is reliably chaining multiple subtasks rather than nudging short-horizon numbers further.
- **Hierarchical reasoning on MimicAnno subtasks**: split a high-level planner that decides "what to do next" from a low-level controller that decides "how to move," using MimicAnno's subtask labels as the bridge.
- **More multimodal inputs**: extend the VLA to consume additional signals beyond images + language — e.g. object detection outputs as an extra conditioning channel.
- **Large-scale pretraining from human demonstrations**: pretrain at scale using robot-shaped data generated from first-person human videos (via the MimicAnno EEF / subtask pipelines), aiming for better generalization to the diversity of everyday assistive actions.

*Hardware side*

- **Less user-specific wearable mechanism**: the current prototype was designed around a single wearer. Future iterations should be easier to put on / take off and less dependent on a specific body.
- **Cross-subject evaluation**: actually evaluate fit, donning / doffing, and long-duration physical burden across different body shapes, rather than a single-user check.

*Runtime side*

- **Lighter on-device inference on Jetson**: push quantization (NF4 / HQQ etc.) and related optimization to reduce latency and memory, making cloud-independent execution more practical.

The longer-term goal is not merely to move a robot arm. It is to keep the assistance close enough to the body that the user can express intent in a few words, let the system adapt from accumulated demonstrations, and let the on-device inference path get faster, lighter, and more natural for daily use.

## Acknowledgements

- **Hackathon**: Kaggle × Google DeepMind Gemma 4 hackathon, 2026-04-02 to 2026-05-18. See [`KaggleArticle.md`](./KaggleArticle.md) for the full write-up.
- **Upstream**: VLA-Adapter (Wang et al., 2025), X-VLA, SigLIP, Gemma 4, LeRobot, SAM3, LIBERO, Open-X-Embodiment, MediaPipe, UniDAC, `faster-whisper`, openWakeWord, Unsloth.

## License

- Root repository: **Apache-2.0** (see [`LICENSE`](./LICENSE))
- Submodules (`MimicRec`, `MimicAnno`, `GEM-4-VLA`): **Apache-2.0**
- Submodule `whisper_hackathon`: **MIT**
- Submodule `raspi_for_vla`: no upstream LICENSE file at the time of writing; redistribution / modification requires coordinating with the upstream author.
- `CAD_Library/` SLDPRT / SLDASM / STEP files: governed by the root license
- Upstream libraries retain their own licenses

### Third-party hardware / SDK

- **reBot Arm B601-DM hardware**: based on [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm), licensed under **CERN-OHL-W-2.0** (CERN Open Hardware Licence Version 2 - Weakly Reciprocal). Any redistribution of the mechanical / electrical design (including derivatives in `CAD_Library/`) must comply with CERN-OHL-W-2.0 in addition to the root Apache-2.0 license that covers the rest of this repository.
- **reBotArm Python control SDK** (`MimicRec/reBotArm_control_py/`, fork of `vectorBH6/reBotArm_control_py`): no explicit upstream LICENSE file. Redistribution or modification of that submodule's source requires coordinating directly with upstream.
- **LeRobot fork** (`MimicRec/lerobot/`, fork of [huggingface/lerobot](https://github.com/huggingface/lerobot)): Apache-2.0 (Hugging Face); incorporates MIT- and Apache-2.0-licensed derived code.
