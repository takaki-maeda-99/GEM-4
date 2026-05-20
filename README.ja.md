# GEM-4

[English](README.md) | **日本語**

> Gemma 4 をバックボーンにしたウェアラブル Vision-Language-Action アシスタント。見て、音声指示を理解し、動いて支援する、ハンズフリーの相棒アームプロトタイプです。

> 本プロジェクトは **Kaggle × Google DeepMind Gemma 4 ハッカソン**（2026-04-02 〜 2026-05-18）の成果物として開発されました。詳細は [`KaggleArticle.md`](./KaggleArticle.md) を参照。

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

### 計算資源とデータの開示

- **学習・評価環境**: 開発および評価には、許可を得て **Toyota Technological Institute・Ukita Lab** の GPU サーバを使用しました。プライベートな研究室データや未公開の研究資産は使用しておらず、公開データセット（OXE、LIBERO）と自分たちで収集したデモンストレーションのみを利用しています。
- **最終デモ環境**: FT checkpoint は約 12 GB（`model.pt` ≈ 11.7 GB）で、市販の 16〜24 GB クラスのコンシューマ GPU、クラウド GPU インスタンス、または Jetson AGX Orin（32 GB）上のオンデバイス実行で動作します。デモの再現に特別な研究室リソースは必要ありません。

## ハードウェア構成部品

ウェアラブル試作機の参考部品リストです。リンクは海外で入手可能な販売元、価格は執筆時点の USD 概算 MSRP です。研究プロトタイプの目安であり、再現用に固定された BOM ではありません（例: RealSense D435i は USB ウェブカムで代替可）。

| 部品 | 用途 | 数量 | リンク | 概算費用 (USD) |
|---|---|---|---|---|
| Intel RealSense D435i | 手首カメラ（USB ウェブカムで代替可） | 1 | [Intel RealSense Store](https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d435i.html) | $329 |
| GoPro HERO11 Black | 胸カメラ本体 | 1 | [gopro.com](https://gopro.com/en/us/shop/cameras/hero11-black/CHDHX-111-master.html) | $400 |
| GoPro Max Lens Mod | HERO11 用広角レンズ | 1 | [gopro.com](https://gopro.com/en/us/shop/mounts-accessories/max-lens-mod/ADWAL-001.html) | $99 |
| GoPro Media Mod | HERO11 用 HDMI / マイク付きフレーム | 1 | [gopro.com](https://gopro.com/en/us/shop/mounts-accessories/camera-media-mod/ADFMD-001.html) | $100 |
| HDMI キャプチャ (USB) | GoPro → Jetson 映像取り込み | 1 | [UGREEN on Amazon](https://www.amazon.com/UGREEN-Capture-Streaming-Recording-Compatible/dp/B0CFQ2BMPZ) | $20 |
| DC-DC コンバータ（uxcell IP68、24 V → 19 V、5 A / 95 W） | ウェアラブル機構の電源（Jetson 19 V レール） | 1 | [uxcell on Amazon](https://www.amazon.com/uxcell-Converter-Regulator-Waterproof-Transformer/dp/B01H97ETVM) | $20 |
| Raspberry Pi 5 (8 GB) | 装着側の音声 / GPIO クライアント（`raspi_for_vla`） | 1 | [raspberrypi.com](https://www.raspberrypi.com/products/raspberry-pi-5/) | $80 |
| NVIDIA Jetson AGX Orin (32 GB H01 Kit) | オンデバイス VLA + Whisper 推論 | 1 | [Seeed Studio](https://www.seeedstudio.com/AGX-Orin-32GB-H01-Kit-p-5569.html) | $1,449 |
| reBot Arm B601-DM + Gripper | ウェアラブル ロボットアーム | 1 | [Seeed Studio](https://www.seeedstudio.com/reBot-Arm-B601-DM-Bundle.html) | $1,197 |
| 3DP パーツ | カスタムマウント（`CAD_Library/` 参照） | — | — | ~$25 |
| MDF・その他材料 | フレーム / ハーネス材料 | — | — | ~$100 |
| **合計** | | | | **≈ $3,819** |

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

**Roadmap, 未実装**

本プロジェクトでは、ウェアラブル VLA システムの中核となる構成要素(データ収集基盤、アノテーションツール、実機インターフェース)を統合しました。一方で、まだ十分に検証できていない部分や、さらに改善が必要な領域も残っています。

第一に、コードベース上は cross-embodiment 学習に対応していますが、複数のロボット形態をまたぐ大規模な検証はまだ行えていません。今後は、異なるロボット間で共有知識・表現がどの程度転移するのかを、よりシステマティックに評価していく必要があります。

第二に、現状の VLA モデルにはまだ改善の余地があります。特に LIBERO ベンチマークの Long タスクに対する性能は限定的であり、複数の subtask を確実につなげる能力は今後の重要な方向性です。

第三に、ウェアラブルハードウェアは依然として試作段階であり、単一ユーザーを前提に設計されています。装着形状や物理的フィットに関するクロス被験者評価はまだ行えていません。今後の評価では、体格差、着脱のしやすさ、長時間使用時の身体的負担を考慮する必要があります。

今後は、人間デモデータから生成したロボットデータを用いた大規模な pretrain を進め、日常環境における多様な支援アクションへの汎化を目指します。あわせて、long-horizon タスクに対する subtask 推論ベースの階層推論の導入や、物体検出結果のようなマルチモーダル入力を取り込めるようモデルを拡張していく予定です。

ハードウェア面では、特定のユーザーへの依存度がより低く、着脱しやすいウェアラブル機構の開発を続けます。さらに、量子化などの最適化技術によって Jetson 上のローカル推論の効率を改善し、latency とメモリ使用量を削減して、クラウド非依存の実行をより実用的なものにしていきます。

## Contributors

| 名前 | 役割 | 貢献 |
|---|---|---|
| [takakimaeda](https://www.kaggle.com/takakimaeda) | リーダー | システム設計、VLA リサーチ・構築、MimicRec 開発、実機制御の各種実装 |
| [halfvolley](https://www.kaggle.com/halfvolley) | メンバー | データ収集、ビデオ撮影協力 |
| [gayagayagaya4](https://www.kaggle.com/gayagayagaya4) | メンバー | アノテーションパイプライン リサーチ、MimicAnno 開発、Jetson 環境構築、音声認識開発（ウェイクワード、Whisper）、QLoRA リサーチ |
| [yutasoyokaze](https://www.kaggle.com/yutasoyokaze) | メンバー | データ収集、実機制御デバッグ |
| [hosakayushun](https://www.kaggle.com/hosakayushun) | メンバー | ハードウェア設計・製作、ビデオ撮影・編集 |

プロフィールリンクは各メンバーの Kaggle アカウントです。

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
