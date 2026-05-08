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

### Representative tasks (operator-supervised scripted demo)

### Sim benchmark

### Reproducibility

## System overview

## Why Gemma 4 + VLA-Adapter

## The stack — 4 components

### `X-VLA-Adapter/` — VLA model & training & inference

### `MimicRec/` — local-first data collection web app

### `MimicAnno/` — offline subtask annotation

### `CAD_Library/` — wearable hardware

## Quickstart

## Safety & Scope

### Current safety measures

### Scope (what the project is, and is not)

## Status & Roadmap

### Shipped

### In progress

### Roadmap — NOT IMPLEMENTED

## Acknowledgements

## License
