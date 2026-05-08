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

### 代表タスク (operator-supervised scripted demo)

### Sim benchmark

### Reproducibility

## System overview

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
