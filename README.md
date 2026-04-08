# HCQ-SFL-IR — Hybrid Classical–Quantum Split Federated Learning for Intrusion Resilience

[![Research Status](https://img.shields.io/badge/status-federated%20aggregation%20phase-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![Dataset](https://img.shields.io/badge/dataset-MIMIC--IV%20v3.1-green)]()
[![License](https://img.shields.io/badge/license-research-lightgrey)]()

> **QR-SecaaS** (Quantum-Resilient Security-as-a-Service) — a privacy-preserving security middleware for federated learning in smart healthcare.

---

## Overview

This repository contains the implementation of **HCQ-SFL-IR**, a Split Federated Learning framework that enables multiple hospitals to collaboratively train an ICU mortality prediction model **without ever sharing raw patient data**. Only compressed 32-dimensional activation vectors (smash layer outputs) leave each client. On the server side, a classification head, quantum-inspired anomaly detection via Parameterised Quantum Circuits (PQCs), and trust-weighted federated aggregation provide intrusion resilience against adversarial attacks.

**Research novelty:** The quadruple intersection of *Split Federated Learning + Intrusion Detection + IoMT Healthcare + Quantum Enhancement* has **zero prior papers** as of early 2026. This is a first-of-its-kind contribution.

### How It Works

```
HOSPITAL A (Medical ICU)        HOSPITAL B (Surgical ICU)       HOSPITAL C (Cardiac ICU)
┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────────┐
│  Raw patient data    │        │  Raw patient data    │        │  Raw patient data    │
│  (266 features)      │        │  (266 features)      │        │  (266 features)      │
│        ↓             │        │        ↓             │        │        ↓             │
│  ClientEncoder       │        │  ClientEncoder       │        │  ClientEncoder       │
│  266 → 64 → 32      │        │  266 → 64 → 32      │        │  266 → 64 → 32      │
│        ↓             │        │        ↓             │        │        ↓             │
│  32-dim activation   │        │  32-dim activation   │        │  32-dim activation   │
└────────┬─────────────┘        └────────┬─────────────┘        └────────┬─────────────┘
         │ only this leaves              │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         ↓
                              ┌──────────────────────┐
                              │   QR-SecaaS SERVER    │
                              │                      │
                              │  Server classifier   │
                              │  PQC anomaly scoring │
                              │  Trust-weighted      │
                              │  FedAvg aggregation  │
                              └──────────────────────┘
```

Raw patient data **never leaves** the client. The privacy guarantee is architectural, not assumed.

---

## Dataset

**MIMIC-IV version 3.1** — real ICU data from Beth Israel Deaconess Medical Center.

| Property | Detail |
|----------|--------|
| Task | Binary mortality prediction (`hospital_expire_flag`) |
| Total ICU stays | ~94,458 |
| Features per stay | 266 |
| Source tables | `icustays`, `admissions`, `patients`, `labevents`, `chartevents`, `inputevents`, `outputevents`, `diagnoses_icd` |

**Feature breakdown (266 total):**

- **Lab features (168):** 24 lab items × 7 statistics (mean, std, min, max, first, last, count)
- **Vital features (84):** 14 vital signs × 6 statistics (mean, std, min, max, first, last)
- **Vasopressor flags (5):** binary indicators for norepinephrine, vasopressin, epinephrine, phenylephrine, dopamine
- **Demographics (6+):** age, gender, insurance, admission type, ethnicity, length of stay
- **Diagnosis:** ICD chapter encoding (first 3 characters of primary ICD code)

### Non-IID Federated Split

Data is split by ICU care unit to create clinically meaningful non-IID distributions — each client has different patient demographics, disease profiles, and mortality rates:

| Client | ICU Units | Stays | Mortality Rate |
|--------|-----------|-------|----------------|
| Client 0 — Medical | MICU, MICU/SICU | ~36,152 | ~16% |
| Client 1 — Surgical | SICU, TSICU | ~23,483 | ~11.5% |
| Client 2 — Cardiac | CCU, CVICU, Neuro ICU | ~32,743 | ~16% |

---

## Client-Side Architecture

Each client runs an identical lightweight MLP encoder. The architecture is consistent with existing SFL-IDS literature for tabular data (ref: AMIA 2023).

### ClientEncoder (19,360 parameters)

| Layer | Input → Output | Operation |
|-------|----------------|-----------|
| Linear 1 | 266 → 64 | Fully connected |
| BatchNorm1d | 64 | Normalise activations |
| ReLU | 64 | Non-linearity |
| Dropout | 64 | p=0.5 |
| Linear 2 | 64 → 32 | Fully connected |
| BatchNorm1d | 32 | Normalise activations |
| ReLU | 32 | **Smash layer output** |

The 32-dimensional output is the only data that crosses the client boundary — an 8.3× compression from the original 266 features that cannot be inverted to recover patient records.

### Privacy-Safe Preprocessing

Preprocessing is performed **inside** each client notebook (not globally) to prevent cross-client data leakage:

1. Load raw CSV (no scaling applied)
2. Train/val split first (80/20, stratified, `random_state=42`)
3. Fit `StandardScaler` on training set only
4. Transform val using training scaler
5. Clip outliers to [-5, 5]
6. Save `scaler.pkl` per client

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | BCEWithLogitsLoss |
| pos_weight | min(neg/pos, 5.0) |
| Optimiser | Adam, lr=1e-4, weight_decay=1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early stopping | patience=8 |
| Max epochs | 80 |

### Client Training Results

| Client | ROC-AUC | PR-AUC | Accuracy | Expired F1 | Best Epoch |
|--------|---------|--------|----------|------------|------------|
| Client 0 — Medical | 0.9585 | — | 93% | 0.77 | 32 |
| Client 1 — Surgical | 0.9700 | 0.8621 | 96% | 0.81 | 34 |
| Client 2 — Cardiac | ✅ Complete | — | — | — | — |

All three clients converge cleanly with train-val loss gaps of ~0.02 (no overfitting).

---

## Repository Structure

```
HCQ-SFL-IR/
│
├── notebooks/
│   ├── MIMIC_IV_SPLIT.ipynb          # Phase 1: Feature engineering + non-IID split
│   ├── client0_model.ipynb           # Client 0 (Medical) encoder training
│   ├── client1_model.ipynb           # Client 1 (Surgical) encoder training
│   └── client2_model.ipynb           # Client 2 (Cardiac) encoder training
│
├── models/                           # Legacy baseline models (exploratory phase)
│   ├── wesad_s2.ipynb
│   ├── pamap2_nn.ipynb
│   ├── mimic3_waveform.ipynb
│   └── ton_iot_model.ipynb
│
├── fl_clients/                       # Generated by MIMIC_IV_SPLIT.ipynb
│   ├── client0_medical.csv
│   ├── client1_surgical.csv
│   ├── client2_cardiac.csv
│   └── models/
│       ├── client0_medical/
│       │   ├── best.pth              # Best checkpoint during training
│       │   ├── encoder.pth           # Encoder weights — used in SFL rounds
│       │   ├── local_model.pth       # Full model backup
│       │   └── scaler.pkl            # StandardScaler (local training data only)
│       ├── client1_surgical/
│       │   ├── best.pth
│       │   ├── encoder.pth
│       │   ├── local_model.pth
│       │   └── scaler.pkl
│       └── client2_cardiac/
│           ├── best.pth
│           ├── encoder.pth
│           ├── local_model.pth
│           └── scaler.pkl
│
├── server/                           # (Next phase)
│   ├── server_model.py               # Server-side classification head
│   ├── fedavg.py                     # FedAvg aggregation
│   ├── pqc_layer.py                  # Parameterised Quantum Circuit module
│   └── trust_aggregator.py           # Trust-weighted federated aggregation
│
├── attacks/                          # (Planned)
│   ├── model_poisoning.py
│   ├── backdoor.py
│   ├── free_rider.py
│   ├── gradient_scaling.py
│   ├── byzantine.py
│   └── sensor_spoofing.py
│
└── README.md
```

> **Note:** `fl_clients/` data and model files are stored on Google Drive and not committed to the repo due to size and patient data restrictions (MIMIC-IV requires credentialed access).

---

## Planned Server-Side Architecture

### Server Classification Head

Receives 32-dim smash layer activations from each client:

```
32 (smash input) → 64 → BatchNorm → ReLU → Dropout → 1 (mortality prediction)
```

### Parameterised Quantum Circuit (PQC) Layer

- Encodes smash layer activations into quantum states
- Computes quantum fidelity and trace distance against reference states
- Activations from poisoned or backdoored clients produce measurably different quantum signatures
- Used for anomaly scoring in the trust-weighted aggregation

### Federated Aggregation

1. **FedAvg baseline** — standard weight averaging across clients
2. **Trust-weighted FedAvg** (QR-SecaaS contribution) — PQC-derived trust scores weight client contributions, conceptually aligned with FLTrust (robust to 40–60% malicious clients)

---

## Attack Simulation Plan

Six categories of adversarial attack, covering all standard FL threats plus two underexplored scenarios:

| Attack | Method | Detection Strategy |
|--------|--------|--------------------|
| Model poisoning | Label flipping on client data | Quantum fidelity deviation from reference |
| Backdoor insertion | Trigger pattern injection | Trace distance anomaly in activations |
| Free-rider | Random/stale activations submitted | Activation distribution divergence |
| Gradient scaling | Gradients scaled by large factor | Norm-based filtering + trust decay |
| Byzantine updates | Arbitrary malicious weight updates | Robust aggregation (Krum/median fallback) |
| Sensor spoofing | Synthetic vital signs injected | Out-of-distribution detection at smash layer |

**Free-rider** and **sensor spoofing** are the two least-explored attacks in FL-IDS literature — covering both strengthens the novelty claim.

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- MIMIC-IV v3.1 access (requires [PhysioNet credentialed access](https://physionet.org/content/mimiciv/3.1/))
- Google Colab (recommended) or local GPU

### Installation

```bash
pip install torch numpy scipy scikit-learn matplotlib pandas seaborn tqdm joblib
```

> Quantum modules (PennyLane / Qiskit) will be added in the server-side phase.

### Running the Pipeline

**Phase 1 — Data preprocessing:**
```
Open notebooks/MIMIC_IV_SPLIT.ipynb in Colab
Mount Google Drive with MIMIC-IV data
Run all cells → produces 3 client CSVs
```

**Phase 2 — Client training (repeat for each client):**
```python
# In notebooks/client{N}_model.ipynb, set:
CLIENT_NAME = 'client0_medical'    # or client1_surgical, client2_cardiac

# Run all cells → saves encoder.pth, scaler.pkl to Drive
```

**Phase 3 — Server + FedAvg:** *(in progress)*

**Phase 4 — Attack simulation:** *(planned)*

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Tabular features** (not raw time-series) | Every SFL-IDS paper in the literature uses tabular/aggregated features — this is the standard |
| **MLP encoder** (not CNN) | Data is tabular (one row per stay); CNN is for image/traffic data. AMIA 2023 also uses FC for tabular EHR |
| **32-dim smash layer** | 8.3× compression (266→32); typical for tabular SFL. Abstract enough to prevent feature reconstruction |
| **Per-client scaling** | Global scaling before splitting leaks cross-client statistics, violating the FL privacy boundary |
| **Non-IID by ICU care unit** | Clinically meaningful heterogeneity — each "hospital" has different patient populations |

---

## Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | MIMIC-IV feature engineering (266 features) | ✅ Complete |
| 1 | Non-IID split by ICU care unit (3 clients) | ✅ Complete |
| 2 | Client-side preprocessing pipeline | ✅ Complete |
| 2 | Client 0 (Medical) encoder training | ✅ Complete |
| 2 | Client 1 (Surgical) encoder training | ✅ Complete |
| 2 | Client 2 (Cardiac) encoder training | ✅ Complete |
| 3 | Server-side classification head | 🔄 In Progress |
| 3 | FedAvg baseline aggregation | 🔄 In Progress |
| 3 | PQC quantum anomaly scoring layer | ⬜ Planned |
| 3 | Trust-weighted federated aggregation | ⬜ Planned |
| 4 | Attack simulation (6 scenarios) | ⬜ Planned |
| 5 | Evaluation and paper writing | ⬜ Planned |

---

## Contributors

**Research Team**
- Students from North South University, BUET, KUET

**Advisor**
- Dr. Sumaiya Tabassum Nimi — Assistant Professor, Dept. of ECE, North South University

**Research Focus:** IoMT Security · Split Federated Learning · Quantum Machine Learning · Privacy-Preserving Healthcare AI

---

## Citation

If you use this work in your research, please cite our upcoming paper (details to be added upon publication).

---

## License

This research project is under active development. License information will be updated upon publication. MIMIC-IV data usage is governed by the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/3.1/).
