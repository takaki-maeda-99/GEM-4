# GEM-4

[English](README.md) | **日本語**

> Gemma 4 をバックボーンにしたウェアラブル Vision-Language-Action アシスタント。見て、音声指示を理解し、動いて支援する、ハンズフリーの相棒アームプロトタイプです。

[![Backbone](https://img.shields.io/badge/Backbone-Gemma%204%20E2B-blue)](https://www.kaggle.com/models/google/gemma-4)
[![Pretrain](https://img.shields.io/badge/Pretrain-OXE%20%2B%20LIBERO%204--suite-blue)](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)
![LIBERO 4-suite avg](https://img.shields.io/badge/LIBERO%204--suite%20avg-74%25-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

![alt text](media/HEROv1.png)

## 概要

棚のコップに手が届かない。フタを開けるのに両手が足りない。ほんの数秒だけ物を支えていてほしい。そうした小さな不自由は、日常の中では何度も現れます。私たちは、その瞬間にそばで動ける「もう一本の腕」を作ろうとしています。

このプロジェクトは、体に装着する 1 本のロボットアームに、胸の俯瞰カメラ、手首カメラ、自然言語の音声指示を組み合わせる試みです。ユーザーが「取って」「開けて」「支えて」と言う。その意図をシステムが読み取り、目の前の状況を見て、短い物理動作として返す。目指しているのは **semi-autonomous な支援**です。完全自律のロボットではなく、人間の意志を拡張する身体の一部に近い存在です。

技術的な核は **SigLIP + Gemma 4-E2B + per-domain projectors + L1 action head** です。Gemma 4 の各層の hidden state に cross-attention する action-generation adapter がアクションを生成し、Gemma 4 本体は凍結したまま、その周りの小さなモジュールだけを学習します。この軽さのおかげで、短いハッカソンスプリントの中でも実験を回し、失敗から戻ることができました。**Open-X-Embodiment + LIBERO を 約 8 : 2** で混ぜた単一の base モデルを pretrain し、そこから per-suite で FT した結果、シミュレーションで **LIBERO 4-suite 平均 74 %**(spatial 72 %、object 92 %、goal 89 %、long 43 %、各 10 episodes × 10 tasks)に到達しています。推論は Jetson 上にも載るため、クラウドに頼らない on-device 実行ができます。

本プロジェクトは operator-supervised の研究プロトタイプであり、医療機器や認証済み支援機器ではありません。

## VLA とは

**Vision-Language-Action (VLA)** は、ロボットのための方策モデルです。次の 3 つをつなぎます。

- **Vision**: カメラに何が見えているか。
- **Language**: 「フタを開けて」「これを支えて」のようなユーザーの指示。
- **Action**: 次にロボットがどう動くか。

このプロジェクトでの VLA は、ユーザーの指示とカメラ画像から、短いロボット動作を出すための橋渡しです。家庭内で単独運用する完全自律ロボットではなく、人間の監督下で動く支援システムの中の learned controller として扱っています。

## できること

- **VLA を end-to-end で学習・評価できる**: [`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE) を **OXE + LIBERO (~8 : 2)** で pretrain し、各 LIBERO suite に FT。Gemma 4 本体は凍結し、projector / action head 側を学習する構成で、FT step 50 k 時点で **LIBERO 4-suite 平均 74 %**(spatial 72 %、object 92 %、goal 89 %、long 43 %、各 suite 100 episodes)。特に Object / Goal は、Gemma 4 の言語・視覚理解を有効活用できていることを示しています。
- **マルチドメインへ広げる土台がある**: GEM-4-VLA は LIBERO だけでなく、複数ドメインの RLDS / LeRobot データと per-domain の input/output projector を扱う前提で設計されています。cross-embodiment / multi-domain 学習も対応スコープに含めており、今後ロボット形態やタスクを増やしていくための中核になります。
- **音声 in、ロボット動作 out、on-device で動く**: `raspi_for_vla` クライアントが「hey GEM」発話をトリガに音声を取得し、Jetson 側の `whisper_hackathon` で文字起こし、その結果と胸 / 手首カメラ画像を VLA `/predict` に渡して動作を返します。マイク・カメラ・文字起こし・推論まですべて on-device に留められ、クラウド往復を必要としません。
- **MimicRec が VLA の入口を作る**: teleop、hand-teach、replay、review、LeRobot v3 export、VLA `/predict` 接続を 1 つの local-first Web アプリにまとめています。ロボット側のインターフェースを抽象化しているので、reBot Arm、SO-101、Isaac Sim、mock のいずれでも、ロボットごとの control adapter を足すだけで同じフローが回ります。デモサイト: <https://takaki-maeda-99.github.io/MimicRec/>。
- **MimicAnno が学習データを濃くする**: subtask boundary は gripper open/close、EEF 速度 / 加速度、action norm の変化から検出。各 segment を SAM3 で追跡し、Unsloth + QLoRA で finetune した Gemma 4 に渡して、verb / object / target / confidence を持つ subtask label を生成します。さらに一人称 GoPro 動画から、MediaPipe の hand landmark と UniDAC の metric depth を組み合わせて EEF 位置・姿勢・pinch 距離を復元するパイプラインも備え、人間デモから直接学習データを生成する道筋を作っています。

<!-- TODO: デモ GIF / 動画を media/ 配下に追加 -->


## 再現情報

| Artifact | Pointer |
|---|---|
| Pretrain ベース | [`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE) — OXE 9 datasets + LIBERO 4-suite mix、`step_100000` |
| FT checkpoints | [`GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial), [`-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object), [`-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal), [`-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10) |
| 学習 config (spatial 例) | [`GEM-4-VLA/configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml`](./GEM-4-VLA/configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml) |
| Eval config (spatial step 50k) | [`GEM-4-VLA/configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml`](./GEM-4-VLA/configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml) |
| Eval コマンド | `uv run python scripts/eval.py configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000.yaml` を `GEM-4-VLA/` で実行 |
| Eval プロトコル | 10 episodes / task × 10 tasks = **suite あたり 100 episodes**、headless MuJoCo |
| FT レシピ | `bs=8 × 2 GPU × accum=2 = eff bs 32` (spatial / object / goal)、`bs=8 × 4 GPU × accum=4 = eff bs 128` (libero_10) |
| Hardware / SW | RTX 6000 Ada / Ubuntu 22.04 / CUDA 12.6 / Python 3.12 / `uv` lockfile |
| 正規化統計 | 各 checkpoint に同梱の `norm_stats.json` |

## システム
<img width="1017" height="712" alt="image" src="https://github.com/user-attachments/assets/6ef4c7d2-b4ad-46ab-9ab7-1a137df98cf5" />

## リポジトリ構成

| Path | Role | Details |
|---|---|---|
| [`GEM-4-VLA/`](./GEM-4-VLA/README.md) | ロボット方策モデル、学習、評価、`POST /predict` 推論サーバ。代表結果は LIBERO 4-suite 平均 74 %(FT step 50 k)。 | [`README`](./GEM-4-VLA/README.md) |
| [`MimicRec/`](./MimicRec/README.md) | teleop、hand-teach、replay、review、LeRobot v3 dataset export を行う local-first Web アプリ。 | [`README`](./MimicRec/README.md) |
| [`MimicAnno/`](./MimicAnno/README.md) | subtask boundary 検出、Gemma 4 VLM labeling、SAM3 tracking、Viterbi smoothing、export のオフラインパイプライン。 | [`README`](./MimicAnno/README.md) |
| [`raspi_for_vla/`](./raspi_for_vla/README.md) | Raspberry Pi 5 側クライアント。USB カメラ・マイクで観測を取得し、GPIO スイッチまたはウェイクワードをトリガに Jetson へ送信、文字起こし結果を受信する。 | [`README`](./raspi_for_vla/README.md) |
| [`whisper_hackathon/`](./whisper_hackathon/README.md) | Jetson AGX Orin 上の Whisper 文字起こしパイプライン（`faster-whisper`）。Raspi から受け取った音声をテキスト化し、VLA 指示として返す。 | [`README`](./whisper_hackathon/README.md) |
| `CAD_Library/` | アーム、グリッパ、ハーネス、カメラ / データ収集治具などのウェアラブル hardware CAD。 | SolidWorks / STEP files via git-lfs |

## Quickstart

```bash
git clone --recurse-submodules <url> && cd vla-gemma-4
git submodule update --init --recursive   # --recurse-submodules を忘れた場合
git lfs install && git lfs pull            # CAD_Library の STEP / SLDASM を取得
```

目的に応じて、各サブモジュールの README に進んでください。

| やりたいこと | 入口 |
|---|---|
| LIBERO で学習 / 評価する | [`GEM-4-VLA/README.md`](./GEM-4-VLA/README.md) |
| ロボットデータを収集 / replay する | [`MimicRec/README.md`](./MimicRec/README.md) |
| 既存 dataset にアノテーションする | [`MimicAnno/README.md`](./MimicAnno/README.md) |

root から直接実行する end-to-end コマンドはまだありません。各サブモジュール README が、それぞれの runtime の SoT です。

**前提環境**: Ubuntu 22.04 / 24.04、Python 3.12、MimicRec frontend 用の Node 20+、GEM-4-VLA 学習用の NVIDIA GPU + CUDA 12.6+。macOS / WSL は一部コンポーネントで動く可能性がありますが未検証です。

## Safety & Scope

- これは **研究プロトタイプ**であり、医療機器や認証済み支援機器ではありません。
- 実機デモは **operator-supervised** であり、物理 E-stop とソフトウェア watchdog が必須です。
- 無監視の家庭内運用は明確にスコープ外です。
- runtime のカメラ / 音声データは既定で on-device に留まり、Jetson 上のオフライン推論ではクラウド送信を行いません。
- 想定する支援は、人間が停止できる状態での短水平・単一タスクのマニピュレーションです。
- accessibility 認証、規制承認、臨床的有効性は主張しません。

## Status & Roadmap

**Shipped**

- GEM-4-VLA v47: FT step 50 k で LIBERO 4-suite 平均 **74 %**(spatial 72 %、object 92 %、goal 89 %、long 43 %)。pretrain base は単一の OXE+LIBERO モデルを共有。
- reBot Arm ウェアラブル上での棚から取る / フタを開ける / 支える の operator-supervised scripted demo。
- reBot Arm、SO-101、Isaac Sim での MimicRec collect / review / replay flow + 公開デモサイト。
- MimicAnno subtask annotation(gripper / EEF シグナル + SAM3 + QLoRA-tuned Gemma 4)と、人間動画から EEF 推定(MediaPipe + UniDAC)。
- 音声パイプライン: `raspi_for_vla`(Pi 5、「hey GEM」ウェイクワード + GPIO)↔ `whisper_hackathon`(Jetson `faster-whisper`)↔ VLA `/predict`。
- VLA `/predict` 契約に対する `xvla_adapter`(実 checkpoint)と `hold_position`(GPU 不要 smoke)の 2 系統 predictor。
- Jetson 上での on-device offline inference。
- `CAD_Library/` のウェアラブル CAD prototype。

**本リポジトリの対応範囲**

- 実 checkpoint での推論は GEM-4-VLA の `serve.py` パスで扱う。FT checkpoint は Hugging Face の `takaki99/GEM-4-FT-*` で公開済み。
- Cross-embodiment / multi-domain は GEM-4-VLA の per-domain projector スコープで扱う(reBot Arm の single-task FT を OXE pretrain base に積む等)。
- Autonomous labeling と edit UI は MimicAnno 側の発展スコープとして扱う。

**Roadmap, not implemented**

この先で目指すのは、単に「ロボットアームが動く」ことではありません。身につけた人が、毎回細かく操作しなくても、自分の意図を短い言葉で伝えられること。環境が少し変わっても、過去の実演と注釈から動作を学び直せること。そして、すでに動き始めている on-device 推論を、より軽く、より速く、より自然な支援へ伸ばしていくことです。

- **データを増やす**: 一人称視点の人間動画から手の動きや把持を推定し、ロボット学習に使える形へ変換する adapter。
- **VLA をスケールする**: LIBERO から実機、単一ロボットから cross-embodiment / multi-domain へ広げ、データと adapter を差し替えながらタスクを増やしていく。
- **長い作業を扱う**: MimicAnno の subtasks を使い、「次に何をするか」を決める planner と、「どう動くか」を実行する low-level controller に分けた hierarchical inference。
- **身につけられる推論をさらに軽くする**: NF4 / HQQ 量子化などで、Jetson 上の on-device 推論をさらに高速・省メモリにする。
- **安全な実世界支援へ進める**: E-stop、watchdog、operator supervision を前提に、短い単発タスクから、より自然な日常動作の連なりへ拡張する。

## Acknowledgements

- **Hackathon**: Kaggle × Google DeepMind Gemma 4 hackathon、2026-04-02〜2026-05-18。詳細は [`KaggleArticle.md`](./KaggleArticle.md) のプロジェクト解説を参照。
- **Upstream**: VLA-Adapter (Wang et al., 2025), X-VLA, SigLIP, Gemma 4, LeRobot, SAM3, LIBERO, Open-X-Embodiment, MediaPipe, UniDAC, `faster-whisper`, openWakeWord, Unsloth.

## License

- root リポジトリ: **Apache-2.0** ([`LICENSE`](./LICENSE) 参照)
- submodule (`MimicRec`, `MimicAnno`, `GEM-4-VLA`): **Apache-2.0**
- submodule `whisper_hackathon`: **MIT**
- submodule `raspi_for_vla`: 上流に LICENSE ファイル未設定。再配布・改変は上流著者との個別調整が必要です。
- `CAD_Library/` 内の SLDPRT / SLDASM / STEP files: root license に従う
- 上流ライブラリは各々のライセンスに従う

### サードパーティ ハードウェア / SDK

- **reBot Arm B601-DM ハードウェア**: [Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) ベース。**CERN-OHL-W-2.0**（CERN Open Hardware Licence Version 2 - Weakly Reciprocal）。機構・回路設計の再配布（`CAD_Library/` 内の派生物を含む）は root の Apache-2.0 に加えて CERN-OHL-W-2.0 にも従う必要があります。
- **reBotArm Python 制御 SDK**（`MimicRec/reBotArm_control_py/`、`vectorBH6/reBotArm_control_py` のフォーク）: 上流に LICENSE ファイル未設定。当該 submodule のソース再配布・改変は上流との個別調整が必要です。
- **LeRobot フォーク**（`MimicRec/lerobot/`、[huggingface/lerobot](https://github.com/huggingface/lerobot) のフォーク）: Apache-2.0（Hugging Face）。一部 MIT / Apache-2.0 派生コードを含みます。
