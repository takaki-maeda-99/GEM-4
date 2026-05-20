# GEM-4

**English** | [日本語](README.ja.md)

> A Gemma 4-based wearable Vision-Language-Action assistant: a hands-free companion arm prototype that sees, understands voice instructions, and moves to help people with limb or visual impairments.

> Built for the **Kaggle × Google DeepMind Gemma 4 Hackathon** (2026-04-02 – 2026-05-18). Full write-up: [`KaggleArticle.md`](./KaggleArticle.md).

[![Backbone](https://img.shields.io/badge/Backbone-Gemma%204%20E2B-blue)](https://www.kaggle.com/models/google/gemma-4)
[![Pretrain](https://img.shields.io/badge/Pretrain-OXE%20%2B%20LIBERO%204--suite-blue)](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)
![LIBERO 4-suite avg](https://img.shields.io/badge/LIBERO%204--suite%20avg-74%25-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

![alt text](media/HEROv1.png)

# Background and Project Overview

Much of daily life assumes that people can freely use both arms, locate objects in front of them, and reach naturally for what they need. For people with visual or upper-limb impairments, this assumption does not hold. This project develops a wearable VLA (Vision-Language-Action) assistant based on Gemma 4 that serves as a “second arm” for these users, together with the data collection, generation, and annotation infrastructure needed to train it as an integrated system.

# Why a wearable form factor

Daily-life assistance often consists of small physical actions: pulling out a chair, opening a door, taking an item from a shelf, picking up a dropped object, retrieving something from a bag, or holding a container steady. These actions are not a fixed task set; they arise unpredictably across everyday situations. Rather than a large robot that automates housework, users need a system that moves with them and assists with the specific action needed at that moment. This motivates a wearable form factor that stays close to the user’s body.

# The challenges this introduces

Wearability increases flexibility but also raises the technical difficulty. Because target actions arise across daily life, hand-engineering a small set of predefined tasks is insufficient. The system must operate safely near the body, handle a constantly shifting first-person camera viewpoint, work within the arm’s limited reachable range, and remain lightweight enough for local execution without relying on a large cloud-hosted model.

# Approach: model and data as two pillars

The project addresses these challenges through both model design and data infrastructure.

On the model side, we built a VLA model that uses Gemma 4’s visual and language understanding to generate robot arm actions from the user’s words and surrounding context. With the lightweight Gemma 4-E2B backbone, the model combines the recognition and action generation capabilities required for daily-life assistance while keeping local execution feasible. Effective learning from diverse data was also a core design requirement.

On the data side, we focused on first-person human demonstration data. Direct teleoperation, collected episode by episode, cannot cover the countless support situations found in daily life. By capturing natural human actions in everyday environments and converting them into robot-training data, we can obtain a broader range of support tasks more naturally and at greater scale. To enable this, we developed a pipeline for converting human demonstrations into robot-training data, MimicAnno for subtask labeling and object annotation, and MimicRec for integrating data collection, management, and inference interfaces. Together, these components form a platform that connects data collection, annotation, training, evaluation, and re-training into a continuous development loop.

# The value of this project

The significance of this project is not simply that it moves a robot arm. It combines a Gemma 4-based wearable VLA architecture that learns from diverse data with a scalable, human-demonstration-driven data generation and training pipeline designed for the diversity of daily-life support tasks. Rather than a one-off demo, we implemented a VLA system that continuously learns from everyday motions and responds to previously unseen support situations.

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

### Compute & Data Disclosure

- **Training / evaluation compute**: Development and benchmarking used a GPU server at **Toyota Technological Institute, Ukita Lab**, with the lab's permission. We did not use any private lab data or unpublished research assets — only public datasets (OXE, LIBERO) and our own collected demonstrations.
- **Final demo compute**: The FT checkpoint is ~12 GB (`model.pt` ≈ 11.7 GB). It runs on commodity hardware — a single 16–24 GB consumer GPU, a cloud GPU instance, or on-device on a Jetson AGX Orin (32 GB) — so no special lab resource is required to reproduce the demo.

## Hardware Bill of Materials

Reference parts list for the wearable prototype. Links point to internationally available vendors and prices are approximate USD MSRPs at time of writing — this is an indicative BOM for a research prototype, not a fixed reproduction recipe, and substitutions (e.g. a USB webcam in place of the RealSense D435i) are expected.

| Component | Use | Qty | Link | Approx. Cost (USD) |
|---|---|---|---|---|
| Intel RealSense D435i | Wrist camera (USB webcam fallback) | 1 | [Intel RealSense Store](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435i.html) | $329 |
| GoPro HERO11 Black | Chest camera body | 1 | [gopro.com](https://gopro.com/en/us/shop/cameras/hero11-black/CHDHX-111-master.html) | $400 |
| GoPro Max Lens Mod | Wide-FOV lens for HERO11 | 1 | [gopro.com](https://gopro.com/en/us/shop/mounts-accessories/max-lens-mod/ADWAL-001.html) | $99 |
| GoPro Media Mod | Frame with HDMI / mic for HERO11 | 1 | [gopro.com](https://gopro.com/en/us/shop/mounts-accessories/camera-media-mod/ADFMD-001.html) | $100 |
| HDMI capture (USB) | GoPro → Jetson video ingest | 1 | [UGREEN on Amazon](https://www.amazon.com/UGREEN-Capture-Streaming-Recording-Compatible/dp/B0CFQ2BMPZ) | $20 |
| DC-DC converter (uxcell IP68, 24 V → 19 V, 5 A / 95 W) | Power regulation for the wearable rig (Jetson 19 V rail) | 1 | [uxcell on Amazon](https://www.amazon.com/uxcell-Converter-Regulator-Waterproof-Transformer/dp/B01H97ETVM) | $20 |
| Raspberry Pi 5 (8 GB) | On-body voice / GPIO client (`raspi_for_vla`) | 1 | [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/) | $80 |
| NVIDIA Jetson AGX Orin (32 GB H01 Kit) | On-device VLA + Whisper inference | 1 | [Seeed Studio](https://www.seeedstudio.com/AGX-Orin-32GB-H01-Kit-p-5569.html) | $1,449 |
| reBot Arm B601-DM + Gripper | Wearable robot arm | 1 | [Seeed Studio](https://www.seeedstudio.com/reBot-Arm-B601-DM-Bundle.html) | $1,197 |
| 3D-printed parts | Custom mounts (see `CAD_Library/`) | — | — | ~$25 |
| MDF & misc materials | Frame / harness materials | — | — | ~$100 |
| **Total** | | | | **≈ $3,819** |

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

In this project, we integrated the core components of a wearable VLA system, including the data collection infrastructure, annotation tools, and real-robot interface. At the same time, there remain aspects that have not yet been fully validated, as well as areas that require further improvement.

First, although cross-embodiment learning is supported in the codebase, we have not yet conducted large-scale validation across multiple robot embodiments. Future work should more systematically evaluate the extent to which shared knowledge and representations can be transferred across different robots.

Second, the current VLA model still has room for improvement. In particular, its performance on Long tasks in the LIBERO benchmark remains limited, and the ability to reliably connect multiple subtasks is an important direction for future work.

Third, the wearable hardware is still at the prototype stage and has been designed around a single user. We have not yet conducted cross-subject evaluations of the wearing form or physical fit. Future evaluations should consider differences in body shape, ease of wearing and removal, and the physical burden of long-term use.

Moving forward, we plan to pursue large-scale pretraining using robot data generated from human demonstration data, with the goal of improving generalization to diverse assistive actions in everyday environments. We also plan to introduce hierarchical reasoning based on subtask inference for long-horizon tasks, and to extend the model to incorporate multimodal inputs such as object detection results.

On the hardware side, we will continue developing a wearable mechanism that is less dependent on a specific user and easier to put on and take off. In addition, we will improve the efficiency of local inference on Jetson through optimization techniques such as quantization, reducing both latency and memory usage and making cloud-independent execution more practical.

## Contributors

| Name | Role | Contributions |
|---|---|---|
| [takakimaeda](https://www.kaggle.com/takakimaeda) | Lead | System architecture, VLA research and implementation, MimicRec development, real-robot control |
| [gayagayagaya4](https://www.kaggle.com/gayagayagaya4) | Member | Annotation pipeline research, MimicAnno development, Jetson environment setup, speech recognition (wake-word, Whisper), QLoRA research |
| [hosakayushun](https://www.kaggle.com/hosakayushun) | Member | Hardware design and fabrication, video shooting and editing |
| [yutasoyokaze](https://www.kaggle.com/yutasoyokaze) | Member | Data collection, real-robot control debugging |
| [halfvolley](https://www.kaggle.com/halfvolley) | Member | Data collection, video shooting support |

Profile links point to each contributor's Kaggle account.

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
