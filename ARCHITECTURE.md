# RVisionT — Architecture Overview

This document describes the architectural design and engineering decisions
behind RVisionT, an engineering-oriented OCR pipeline.

The goal of the architecture is not maximum OCR accuracy, but **robustness,
stability, and controlled search complexity** when working with unstable OCR engines.


## High-Level Design

RVisionT implements a **two-phase OCR pipeline** with adaptive preprocessing,
heuristic scoring, and explicit failure handling.

Core principles:

* reduce OCR search space
* avoid unbounded brute-force
* isolate unstable operations
* prefer stability over raw accuracy


## Pipeline Overview

```
Input Image
     │
     ▼
Preprocessing + Normalization
     │
     ▼
Phase A — Orientation & Baseline Search
     │
     ▼
Best Orientation Selected
     │
     ▼
Phase B — Focused OCR Refinement
     │
     ▼
Optional ROI-Based OCR Fallback
     │
     ▼
Post-processing & Cleanup
     │
     ▼
Final Text Output
```


## Phase A — Orientation & Baseline Search

### Purpose

Phase A determines:

* correct image orientation
* baseline OCR configuration

This phase prioritizes **speed and safety**.

### Strategy

Phase A iterates over:

* image rotations (0°, 90°, 270°)
* a limited set of Tesseract PSM modes
* language configurations (rus, eng, rus+eng)

Key design choices:

* uses `image_to_string` only (lower memory usage)
* avoids aggressive thresholding
* applies early stopping once a confident result is found

### Output

Phase A produces:

* best rotation
* baseline OCR result
* metadata describing the chosen configuration

## Phase B — Focused OCR Refinement

### Purpose

Phase B refines OCR quality using the **best orientation selected in Phase A**.

This phase trades speed for quality but within a controlled search space.

### Strategy

* runs only on a single orientation
* evaluates multiple preprocessing variants
* selectively uses `image_to_data` to estimate confidence
* computes a composite heuristic score for each candidate

Scoring factors include:

* text length
* average confidence
* character distribution
* noise and garbage penalties


## ROI-Based OCR Fallback

### Motivation

Full-frame OCR often performs poorly on:

* meme-style images
* captions located at top/bottom regions

### Approach

A lightweight text-band detection heuristic is applied:

* horizontal density analysis
* band merging and filtering
* top/bottom caption heuristics

Each detected region is processed independently using Phase B logic,
and results are merged top-to-bottom.

This fallback is applied **only when layout heuristics indicate captions**.


## Preprocessing Architecture

Preprocessing is **variant-based**, not fixed.

Examples:

* grayscale / inverted grayscale
* unsharp masking
* adaptive thresholding (Phase B only)
* bilateral filtering

Each preprocessing variant is treated as a candidate and scored independently.

This avoids committing to a single preprocessing strategy.


## Heuristic Scoring System

RVisionT uses a fully heuristic quality scoring system due to the absence
of labeled training data.

Key scoring components:

* character composition (letters vs noise)
* script consistency (Cyrillic vs Latin)
* word structure analysis
* confidence weighting (when available)
* penalties for OCR artifacts

This system prioritizes **readable text over longer but noisy output**.


## Stability & Failure Handling

OCR engines such as Tesseract are treated as **unstable dependencies**.

Explicit safeguards include:

* downscaling large images before OCR
* skipping unsafe image sizes
* limiting `image_to_data` usage
* strict timeouts on OCR calls
* catching and isolating OCR engine failures

At no point does a single OCR failure terminate the pipeline.


## Architectural Trade-offs

### Chosen Trade-offs

* Heuristics instead of ML models
* Controlled search instead of exhaustive brute-force
* Stability over maximum accuracy
* Readability over raw text length

### Intentionally Not Addressed

* deep learning text detection
* production-grade APIs
* large-scale OCR benchmarking
* language model post-correction

These are considered future extensions, not architectural goals.

## Extensibility

The architecture allows future additions such as:

* learned scoring functions
* external text detectors (EAST, CRAFT)
* language-specific post-processing modules
* GUI or service wrappers


## Summary

RVisionT demonstrates a **robust OCR pipeline architecture**
designed around real-world OCR instability and noisy input data.

The project emphasizes:

* systematic engineering thinking
* explicit failure control
* adaptive processing pipelines
* informed architectural trade-offs
