# SignFlow – Landmark‑Based ASL Model Trainer

This repository is a **sub‑project of SignFlow** focused on one very specific problem:

> **Training a high‑accuracy ASL classifier from scratch using hand landmarks instead of images.**

---

## Why this exists

Most public ASL models:

* operate on raw images
* require huge datasets
* rely on rule systems
* generalize to real hands, real lighting, and real cameras

This project takes a different route:

**MediaPipe → landmarks → engineered features → classical ML classifier**

The result:

* very small latency
* extremely high accuracy
* fast training
* explainable behavior
* works in real time

The model improves continuously as you add samples.

---

## High‑level architecture

```
Browser (MediaPipe Hands)
        │
        │ 21 landmarks per hand (x, y, z)
        ▼
Feature Engineering (client‑side JS)
        │
        │ 146‑dimensional feature vector
        ▼
Flask API (/save_sample)
        │
        │ CSV dataset
        ▼
Offline Training (train_model.py)
        │
        ▼
Serialized Model (model.pkl)
```

---

## Core idea: landmarks, not pixels

MediaPipe Hands detects **21 landmarks per hand** in 3D space.

Each frame becomes structured data instead of raw pixels. This removes:

* lighting variance
* background noise
* camera resolution dependency

What remains is **pure hand geometry**.

---

## Feature representation

Each hand produces **73 features**:

### 1. Normalized coordinates (63)

* 21 landmarks × (x, y, z)
* All landmarks are:

  * translated so landmark 0 (wrist) is the origin
  * scaled using landmark 9 as a reference

This makes features:

* scale‑invariant
* position‑invariant
* camera‑distance‑invariant

### 2. Finger joint angles (10)

Angles are computed between consecutive finger joints.

This captures:

* finger curl
* finger extension
* subtle differences between similar signs (M / N / S / T / A)

Angles give the model *structure awareness*.

---

## Two‑hand support

Each frame stores:

* **Primary hand features** (73)
* **Secondary hand features** (73)

Total feature vector length:

```
73 + 73 = 146
```

If only one hand is visible:

* the missing hand is zero‑filled
* a flag `only_primary_hand` is stored

This lets the model learn:

* one‑hand vs two‑hand signs
* symmetric vs asymmetric gestures

---

## Dataset format

Samples are appended to a CSV file:

```
label, only_primary_hand,
primary_x0, primary_y0, primary_z0, ... , primary_angle9,
secondary_x0, secondary_y0, secondary_z0, ... , secondary_angle9
```

## Client‑side pipeline (browser)

MediaPipe runs **entirely in the browser**.

Per frame:

1. Camera frame → MediaPipe Hands
2. Landmarks extracted
3. Normalization + angle computation
4. Feature vector built
5. Feature vector sent to server (JSON)

## Flask API

### `/save_sample` (POST)

Receives:

* `features`: length‑146 float array
* `label`: ASL class (A–Z, HOME, I_LOVE_YOU, etc.)
* `dataset`: CSV name
* `only_primary_hand`: boolean flag

Validates and appends the sample to disk.

---

## Model training

Training is performed offline using `train_model.py`.

Typical workflow:

1. Load CSV dataset
2. Split train / test
3. Train a multi‑class classifier
4. Evaluate precision / recall / F1
5. Serialize model

Example result:

```
accuracy: ~95–100%
classes: 24 alphabets + HOME + I_LOVE_YOU
training time: ~10 seconds
model size: ~6 MB
```

---

## How this fits into SignFlow

This sub‑project solves:

> “How do we reliably classify ASL signs in real time?”

The main SignFlow project adds:

* networking
* video calls
* streaming predictions
* multi‑user sessions

This model is the brain.

---

## Status

* Dataset collection: complete
* Model accuracy: excellent
* Real‑time inference: stable
* Next step: Integration into SignFlow!

---