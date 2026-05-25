# 📦 Warehouse Routing Intelligence (WRI)

> A fully client-side, zero-dependency AI system for predicting warehouse routing destinations from parcel tracking IDs — built entirely in a single HTML file.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works — The Three-Layer Cascade Engine](#how-it-works--the-three-layer-cascade-engine)
  - [Layer 1: Deterministic Rule Engine](#layer-1-deterministic-rule-engine)
  - [Layer 2: Merchant Lookup Table](#layer-2-merchant-lookup-table)
  - [Layer 3: Gradient Boosted Trees (GBT)](#layer-3-gradient-boosted-trees-gbt)
  - [Cascade Decision Logic](#cascade-decision-logic)
- [AI Model — Deep Dive](#ai-model--deep-dive)
  - [Feature Engineering](#feature-engineering)
  - [Training Process](#training-process)
  - [Model Serialization & Loading](#model-serialization--loading)
- [Multi-Device Session System](#multi-device-session-system)
- [User Interface](#user-interface)
- [Getting Started](#getting-started)
- [CSV Training Data Format](#csv-training-data-format)
- [Tech Stack](#tech-stack)
- [Architecture Diagram](#architecture-diagram)
- [Browser Compatibility](#browser-compatibility)
- [License](#license)

---

## Overview

**Warehouse Routing Intelligence** is a single-file web application that uses a three-layer machine learning cascade to predict which warehouse a parcel belongs to based solely on its tracking ID (and optionally its merchant name). It is designed for warehouse scanning workflows where fast, accurate routing decisions are critical.

The entire AI pipeline — training, inference, serialization, and UI — runs **100% in the browser**. No server, no backend, no external ML library. You bring a CSV of historical routing data; WRI learns from it and starts classifying in real-time.

---

## Features

- **🧠 Three-layer AI cascade** — rule engine → merchant lookup → gradient-boosted tree ensemble
- **📋 Train from CSV** — upload any CSV with `trackingId, merchant, warehouse` columns to train a model on the fly
- **💾 Save & load models** — export trained models as JSON files and reload them instantly
- **📡 Multi-device real-time sessions** — link a desktop hub to mobile scanner devices over peer-to-peer WebRTC (via PeerJS)
- **📱 Mobile & desktop modes** — responsive UI with a dedicated mobile scanning view and fullscreen result overlay
- **🔗 QR code join** — generate a scannable QR code so mobile devices can auto-join a session
- **🌗 Dark/light theme** — toggleable via floating action button
- **📊 Live analytics** — per-warehouse and per-merchant scan counters with progress bars
- **🔁 No internet required after load** — fully offline capable once the page is loaded

---

## How It Works — The Three-Layer Cascade Engine

Every tracking ID lookup passes through up to three decision layers. The engine returns as soon as any layer produces a confident result, falling through to the next only when needed.

```
Input ID
   │
   ▼
┌─────────────────────────────┐
│  Layer 1: Rule Engine       │  ~65% coverage, ~98% accuracy
│  (deterministic patterns)   │──────────────────► Result (if certain)
└─────────────┬───────────────┘
              │ uncertain / not covered
              ▼
┌─────────────────────────────┐
│  Layer 2: Merchant Lookup   │  Pure merchants → certain
│  (lookup table from CSV)    │──────────────────► Result (if pure merchant)
└─────────────┬───────────────┘
              │ mixed merchant or unknown
              ▼
┌─────────────────────────────┐
│  Layer 3: GBT Ensemble      │  Handles ambiguous POZ1 vs POZ2 splits
│  (trained on feature vecs)  │──────────────────► Result + confidence
└─────────────────────────────┘
              │ GBT + soft merchant signal blend
              ▼
           Final prediction: { warehouse, confidence, resolvedBy }
```

---

### Layer 1: Deterministic Rule Engine

Covers approximately **65% of all IDs** with roughly **98% accuracy** using hard-coded structural patterns derived from data analysis.

**20-digit numeric IDs (DHL/GLS format)**
- Inspects the sort code at positions 8–11 and 4–7
- Known BER3 sort codes: `7577`, `5249`, `5192`, `5139`, `7137`, `5140`, `9213`, `7824`, `0609`
- Known POZ sort codes: `0734`, `5320`, `7760`, `8803`, `8830`, `9247`, `8859`, `8852` (passed to GBT for POZ1 vs POZ2 disambiguation)
- Prefix matching on first 10 characters for additional certainty

**11-digit numeric IDs**
- Prefix `1155` → BER3 (99% confidence)
- Prefix `115` → BER3 (88% confidence)

**International format** (`[2 letters][digits][2 letters]`, e.g. `CD123456DE`)
- Prefixes `CD`, `LE`, `CB`, `UE`, `CM` → BER3 (96% confidence)
- Prefixes `CA`, `UF` → passed to GBT (POZ split)

**`(Y)` internal system codes**
- `(Y)#[A-Z]{2}...` (without `SRDE`) → BER3 (98% confidence)
- Contains `SRDE` → POZ, passed to GBT
- Other `(Y)[A-Z0-9]` codes → BER3 (85% confidence)

**Explicit prefix rules**
- `LE\d` → BER3 (96%)
- `SCC` → BER3 (99%)
- `CD3` → BER3 (99%)
- `9L3` → POZ1 (99%)

If the rule engine returns `certain: false`, the ID falls through to Layer 2.

---

### Layer 2: Merchant Lookup Table

Built at training time from the CSV data. Merchants are categorized as:

- **Pure merchants** — 100% of their historical parcels routed to a single warehouse. 162 out of 190 merchants fall into this bucket. Lookups are deterministic (confidence = 1.0).
- **Mixed merchants** — split routing history, typically between POZ variants. 28 merchants fall here. Their data becomes a soft signal blended with Layer 3.

When a pure merchant match is found, the result is returned immediately. Mixed merchant data is passed as a weighted prior to the GBT model in Layer 3.

---

### Layer 3: Gradient Boosted Trees (GBT)

A **pure-JavaScript GBT implementation** with no external ML dependencies. This layer handles the hardest cases — particularly disambiguating **POZ1 vs POZ2** routing where IDs share similar structural patterns.

**Architecture:**
- Multi-class classifier (one tree ensemble per output class, OvR style)
- Ensemble of `N` boosting rounds, each round adds one tree per class
- Leaf values represent soft probability distributions (not hard labels)
- Final prediction uses **softmax** over summed logits across all trees
- Learning rate: `0.25`
- Training uses a **15% validation split** for accuracy reporting

**Tree construction:**
- Gini impurity criterion for split selection
- Random feature subsampling per node: `max(4, floor(sqrt(n_features)))` features considered per split
- Up to 10 candidate split thresholds per feature (midpoints between unique values)
- Configurable `maxDepth` and `minSamples` per leaf

---

## AI Model — Deep Dive

### Feature Engineering

Each tracking ID is converted to a **fixed-length feature vector** (Float32Array) of ~42 dimensions:

| Group | Features | Description |
|-------|----------|-------------|
| Format type | 6 | One-hot: `NUM20`, `NUM11`, `INTL`, `Y_HASH_ALPHA`, `Y_HASH_NUM`, `Y_OTHER` |
| ID length | 1 | Normalised to `[0, 1]` over range 0–30 |
| Positional char codes | 14 | Character codes at positions 0–13, normalised by 127 |
| 4-char window hashes | 4 | Rolling hash of 4-char windows at positions 0, 4, 8, 12 |
| Prefix hashes | 5 | Hash of first 4, 6, 8, 10, 12 characters |
| Suffix hashes | 2 | Hash of last 2 and last 4 characters |
| Character ratios | 3 | Digit ratio, alpha ratio, special-char ratio |
| Positional bigrams | 9 | Binary features for known discriminative position+bigram combos |

**Hashing function (32-bit polynomial rolling hash):**
```
h = 0
for each char c: h = (h * 31 + charCode(c)) & 0xFFFF
normalized = h / 65535
```

**Discriminative positional bigrams used:**
`0:00`, `4:04`, `8:07`, `8:05`, `8:08`, `0:LE`, `0:CD`, `0:CA`, `0:CB`

---

### Training Process

Training is triggered by uploading a CSV file. The full pipeline runs asynchronously in the browser:

1. **CSV parsing** — extracts `{ trackingId → { merchant, warehouse } }` mapping
2. **Stat building** — constructs prefix maps, n-gram stats (2–4), Markov suffix stats, length stats, positional bigram stats
3. **Merchant table building** — classifies each merchant as pure or mixed
4. **Feature matrix construction** — calls `extractFeatures(id)` for every training ID
5. **Label encoding** — maps warehouse strings to class indices
6. **Train/validation split** — 15% held out for accuracy estimation
7. **GBT training loop** — iterates boosting rounds, fitting one shallow tree per class per round on the residuals
8. **Validation accuracy** — computed on held-out split and displayed in the UI
9. **Auto-save** — completed model is serialised and downloaded as `warehouse_model.json`

A **progress bar** in the sidebar tracks training completion in real-time.

---

### Model Serialization & Loading

Trained models are saved as JSON with the following schema:

```json
{
  "version": 2,
  "savedAt": "<ISO timestamp>",
  "gbtClasses": ["BER3", "POZ1", "POZ2"],
  "gbtAccuracy": 0.94,
  "gbt": [ /* serialised tree ensemble */ ],
  "merchantStats": { "<merchant>": { "<warehouse>": <count> } },
  "pureMerchants": { "<merchant>": "<warehouse>" },
  "mixedMerchants": { "<merchant>": true },
  "learnedPatterns": { /* prefix map */ },
  "nGramStats": { /* n-gram stats */ },
  "markovStats": { /* suffix stats */ },
  "lengthStats": { /* length→warehouse counts */ },
  "positionalStats": { /* positional bigram stats */ },
  "ruleStats": { /* prefix and positional window stats */ }
}
```

**Tree nodes are compactly serialised:**
- Internal node: `{ l: 0, f: <featIdx>, t: <threshold>, L: <leftNode>, R: <rightNode> }`
- Leaf node: `{ l: 1, v: <probabilityArray> }`

**Version compatibility:**
- `version >= 2` loads the full GBT ensemble
- `version 1` (legacy) restores only stat layers (no GBT); merchant tables are rebuilt from raw stats

---

### Cascade Decision Logic (Detailed)

```javascript
hybridPredict(id, merchantHint):
  1. Exact match in training data?          → return immediately, confidence 1.0
  2. Rule engine returns certain result?    → return, rule confidence
  3. Merchant is pure?                      → return, confidence 1.0
  4. GBT model available?
     a. Merchant has soft signal?
        → blend: GBT probs × 0.75 + merchant probs × 0.35, softmax
     b. No merchant signal?
        → return GBT result directly
  5. Merchant soft signal only (no GBT)?   → return, confidence × 0.7
  6. No signal at all?                      → return "Unknown", confidence 0
```

---

## Multi-Device Session System

WRI uses **PeerJS** (WebRTC data channels via `0.peerjs.com`) to connect multiple devices in real-time.

**Session roles:**
- **Host (desktop)** — creates a 6-character alphanumeric session code, acts as the relay hub
- **Scanner (mobile)** — joins by entering the code or scanning a QR code

**Session message protocol:**
| Message type | Direction | Payload |
|---|---|---|
| `hello` | Scanner → Host | `{ label, deviceId }` |
| `welcome` | Host → Scanner | `{ sessionCode }` |
| `scan` | Any → Host → All | `{ id, wh, conf, merchant, deviceId, deviceLabel, ts }` |
| `history` | Host → new Scanner | `{ scans: last 20 }` |
| `bye` | Scanner → Host | — |

**QR code join flow:**
The QR code encodes a URL with `?join=<CODE>`. Opening that URL on a mobile device automatically switches to mobile view, pre-fills the code, and initiates the join handshake.

**Session capacity:** Up to 60 scans are retained in the shared session history.

---

## User Interface

**Desktop layout (two-column dashboard):**
- **Sidebar** — model state panel, train/load controls, session controls, live device list, per-warehouse stats, inline QR code panel
- **Main area** — tracking ID lookup bar, result panel (warehouse badge + confidence + layer details), recent scans table, merchant summary table

**Mobile layout:**
- Fullscreen result overlay with large warehouse name (colour-coded), confidence bar, and auto-dismiss timer ring
- Counter strip showing BER / POZ / Total
- Mobile join screen on first load

**Confidence colour coding:**
| Badge | Threshold |
|-------|-----------|
| HIGH (green) | ≥ 80% |
| MED (amber) | ≥ 55% |
| LOW (orange) | ≥ 35% |
| VERY LOW (red) | < 35% |

**Warehouse colour coding:**
- 🟢 `BER*` — green
- 🔴 `POZ*` — red
- 🟡 `Unknown` — amber
- 🔵 Other — blue

---

## Getting Started

### 1. Open the app

Just open `index.html` in any modern browser. No build step, no npm, no server.

```bash
# Option A: direct file open
open index.html

# Option B: simple local server (avoids any browser file:// quirks)
python3 -m http.server 8080
# then visit http://localhost:8080
```

### 2. Prepare training data

Create a CSV file with at least three columns:

```
trackingId,merchant,warehouse
00340434756781234567,Acme GmbH,BER3
00340435073219876543,Beta Store,POZ1
LE123456789DE,Gamma Ltd,BER3
```

The header row is auto-detected (any header containing `trackingid`, `tracking_id`, or `id`).

### 3. Train a model

1. Click **▶ Train** in the sidebar to expand the train panel
2. Upload your CSV file
3. Watch the progress bar — training runs in the browser
4. The model auto-saves as `warehouse_model.json` when done

### 4. Or load a saved model

Click **Load model** and select a previously saved `warehouse_model.json`.

### 5. Start scanning

Type or scan (via barcode scanner keyboard-wedge) a tracking ID into the lookup bar and press **Enter** or click **Lookup**.

### 6. Multi-device (optional)

- On desktop: click **＋ New session** to get a 6-letter code
- On mobile: open the same `index.html`, enter the code, or scan the QR code shown in the sidebar

---

## CSV Training Data Format

| Column | Required | Notes |
|--------|----------|-------|
| `trackingId` | ✅ | Raw tracking ID string, any format |
| `merchant` | ✅ (can be blank) | Merchant/seller name — used for Layer 2 |
| `warehouse` | ✅ | Target label, e.g. `BER3`, `POZ1`, `POZ2` |

- Quoted fields and embedded commas are handled correctly
- Header row is optional but recommended
- Minimum 5 rows required to train GBT; fewer rows train stat layers only

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| UI framework | Vanilla HTML/CSS/JS (no framework) |
| Fonts | JetBrains Mono, Syne (Google Fonts) |
| P2P networking | [PeerJS 1.5.4](https://peerjs.com/) over WebRTC |
| QR code generation | [qrcodejs 1.0.0](https://github.com/davidshimjs/qrcodejs) |
| ML engine | Custom pure-JS GBT implementation |
| Storage | In-browser (`Blob` + `FileReader`); model exported as `.json` |
| Runtime dependencies | Zero (ML, UI, session logic all self-contained) |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        index.html                           │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │   CSV Upload │───►│       Training Pipeline          │  │
│  └──────────────┘    │  parseCsv → buildRuleStats       │  │
│                       │  buildMerchantStats              │  │
│  ┌──────────────┐    │  buildNGramStats                 │  │
│  │  Model Load  │    │  trainGbtModel → saveModel       │  │
│  │  (.json)     │    └──────────────┬───────────────────┘  │
│  └──────┬───────┘                   │                       │
│         │            ┌──────────────▼───────────────────┐  │
│         └───────────►│        In-Memory Model State     │  │
│                       │  ruleStats, merchantStats        │  │
│                       │  pureMerchants, mixedMerchants   │  │
│                       │  gbtModel[], gbtClasses[]        │  │
│                       └──────────────┬───────────────────┘  │
│                                      │                       │
│  ┌──────────────┐    ┌──────────────▼───────────────────┐  │
│  │  Tracking ID │───►│     hybridPredict(id, merchant)  │  │
│  │  Input       │    │  Layer 1 → Layer 2 → Layer 3     │  │
│  └──────────────┘    └──────────────┬───────────────────┘  │
│                                      │                       │
│  ┌──────────────┐    ┌──────────────▼───────────────────┐  │
│  │  Result UI   │◄───│  { warehouse, confidence,        │  │
│  │  + Session   │    │    resolvedBy, layerDetails }    │  │
│  │  Broadcast   │    └──────────────────────────────────┘  │
│  └──────────────┘                                           │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │        PeerJS Session Layer (WebRTC)                  │  │
│  │  Host ◄──────────────────────────────► Scanner(s)    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome / Edge (v90+) | ✅ Full support |
| Firefox (v88+) | ✅ Full support |
| Safari (v15+) | ✅ Full support |
| Mobile Chrome/Safari | ✅ Full support |

> WebRTC (PeerJS sessions) requires an internet connection to the STUN/signalling server on first connect, but all ML inference works fully offline.

---

## License

MIT License — free to use, modify, and distribute.
