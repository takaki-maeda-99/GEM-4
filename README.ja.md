# vla-gemma-4

[English](README.md) | **日本語**

> Gemma 4 をバックボーンとした、ハンズフリー・視覚共有のウェアラブル相棒アーム — 腕や視覚に不自由のある方の日常を補助するプロトタイプ。

[![Hackathon](https://img.shields.io/badge/Hackathon-Gemma%204%20Good-orange)](https://www.kaggle.com/competitions/gemma-4-good-hackathon/)
[![Backbone](https://img.shields.io/badge/Backbone-Gemma%204%20E2B-blue)](https://www.kaggle.com/models/google/gemma-4)
![Sim benchmark](https://img.shields.io/badge/LIBERO--Spatial-94%25-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

<!-- TODO: docs/images/hero.gif — 装着デモ ヒーロー -->

## TL;DR

- **このプロジェクト**: 音声で話しかけ、行動で応える、Gemma 4 ベースのウェアラブル VLA アシスタント。
- **なぜ Gemma 4 + VLA-Adapter か**: PLE × VLA-Adapter の bridge attention は他に類を見ないアーキ整合性をもち、 LLM 本体は凍結、 小さな adapter / projector / action head のみを学習する → 短期間での効率的な適応。 E2B クラスのオープンウェイト LLM だから on-device オフライン推論 (Jetson) も射程に入れた。
- **現在動く範囲**: シミュレーションで LIBERO-Spatial 94 % (X-VLA-Adapter v33)、 加えて実機 3 タスク (棚から取る / フタを開ける / 支える) の operator-supervised scripted demo。
- **5 分で評価するなら**:
  1. デモのメディアを見る → [Demo / What it does today](#demo--what-it-does-today)
  2. アーキを読む → [System overview](#system-overview)
  3. [Why Gemma 4 + VLA-Adapter](#why-gemma-4--vla-adapter) を流し読み
  4. LIBERO 94 % の根拠を確認 → [Reproducibility](#reproducibility)

## Mission

腕や視覚に不自由のある方は、 「もう一本の腕」 を必要としている — 棚からコップを取りたい、 フタを開けたい、 物を支えていてほしい、 そういう日常の所作を、 必要なタイミングで頼める腕を。

私たちが作っているのはまさにそれ。 体に装着できる 1 本のアーム、 胸の俯瞰カメラと wrist カメラ、 自然言語の音声指示を受け取り、 視覚をユーザーと共有しながら物理世界に作用する。 設計思想は **semi-autonomous な相棒** — 完全自律ではなく、 ユーザーの意図を拡張し、 補助する。

なぜ今か。 Gemma 4 のオープンウェイト・小型・on-device 性能と、 VLA-Adapter (Wang et al., 2025) の LLM 凍結 + 小さな adapter のみ学習する手法が組み合わさって、 **短期間・効率的な適応**が現実的射程に入った。 約 1.5 ヶ月のハッカソンスプリントでも LIBERO-Spatial 94 % まで到達できる。 *これは operator-supervised の研究プロトタイプであり、 医療機器ではありません。*

## Demo / What it does today

> **状態の注釈**: 以下の実機デモはすべて **operator-supervised の scripted demo** です。 物理 e-stop とソフトウェア watchdog の併用がセッション必須前提。 GIF / 動画はファイルが揃い次第差し替えます。

### 代表タスク (operator-supervised scripted demo)

- **棚から取る** — <!-- TODO: docs/images/demo_shelf.gif -->
- **フタを開ける** — <!-- TODO: docs/images/demo_lid.gif -->
- **支える / 持つ** — 持続的アシスト。 一般的な VLA pick-and-place との差別化点 — <!-- TODO: docs/images/demo_hold.gif -->

### Sim benchmark

- **LIBERO-Spatial 94 %** — X-VLA-Adapter v33、 47 / 50 episodes、 50 episode/suite、 4 タスク × 50 step
- 学習曲線: <!-- TODO: docs/images/v33_training_curve.png -->

### Reproducibility

| Artifact | Pointer |
|---|---|
| 学習 config | [`X-VLA-Adapter/configs/train/libero_spatial_v33.yaml`](./X-VLA-Adapter/configs/train/libero_spatial_v33.yaml) |
| Eval config | [`X-VLA-Adapter/configs/eval/libero_v33_step40000.yaml`](./X-VLA-Adapter/configs/eval/libero_v33_step40000.yaml) |
| Checkpoint | <!-- TODO: HF Hub または Drive 直リンク --> |
| Eval コマンド | `uv run python scripts/eval.py configs/eval/libero_v33_step40000.yaml` (`X-VLA-Adapter/` 配下で実行) |
| ハードウェア / SW | RTX 6000 Ada / Ubuntu 22.04 / CUDA 12.6 / Python 3.12 / `uv` lockfile committed |
| 正規化統計 | checkpoint と同梱 (`norm_stats.json`) |
| Model card | <!-- TODO: 限界・既知の挙動を含む model card へのリンク --> |

## System overview

> 凡例: 緑 solid = shipped / 黄 dot-dash = in-progress (stub / smoke / 学習中) / 橙 dashed = planned / 青 = hardware / 紫 = data store / ピンク = actor。
> Edge: 実線 = 動作する経路、 点線 = in-progress または planned へ向かう経路。

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

主要なデータ流: (1) ユーザーの音声 → MimicRec (推論 client) → X-VLA-Adapter Phase 0 server、 (2) 胸の俯瞰カメラ + wrist カメラ → SigLIP → VLA-Adapter bridge attention で Gemma 4 (frozen) と対話 → action head → 1 アーム、 (3) データ収集ループ: MimicRec で集めた LeRobot v3 episode → MimicAnno で subtask アノテ → X-VLA-Adapter で再学習。 on-device offline 動作 (Jetson クラス) を **目標** とし、 学習はクラウド/オンプレ GPU。

<!-- 任意: docs/images/system_architecture.png — より洗練された装着写真ベースの図、 用意でき次第差し替え -->

## Why Gemma 4 + VLA-Adapter

## The stack — 4 components

### `X-VLA-Adapter/` — VLA モデル / 学習 / 推論

### `MimicRec/` — local-first データ収集 Web アプリ

### `MimicAnno/` — オフライン subtask アノテーション

### `CAD_Library/` — ウェアラブルハードウェア

## Quickstart

## Safety & Scope

### 現状の安全措置

### Scope (何であって何でないか)

## Status & Roadmap

### Shipped

### In progress

### Roadmap — NOT IMPLEMENTED

## Acknowledgements

## License
