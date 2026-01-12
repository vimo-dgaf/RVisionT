# RVisionT
### Engineering-Oriented OCR Pipeline (Two-Phase, Stable)

RVisionT is an experimental Optical Character Recognition (OCR) project built with
Python, OpenCV, and Tesseract OCR.

The project focuses on **engineering robustness**, **stability**, and **adaptive OCR configuration**
rather than maximum accuracy or end-to-end deep learning models.


## What this project demonstrates

- Design of a multi-stage OCR pipeline
- Robust handling of unstable third-party OCR engines (Tesseract)
- Heuristic-based quality scoring and early stopping
- Trade-offs between accuracy, performance, and stability
- Computer Vision preprocessing without deep learning


## What this project is NOT

- Not a production-ready OCR system
- Not a deep learning OCR framework
- Not optimized for maximum recognition accuracy

This project is intentionally positioned as a **research and engineering showcase**.

## Problem Statement

Vanilla Tesseract OCR often fails on:
- meme-style images
- mixed-language content (Russian + English)
- large screenshots
- noisy or stylized text

Common issues include:
- unstable results
- excessive garbage output
- memory crashes (std::bad_alloc)


## Solution Overview

RVisionT implements a **Two-Phase Stable OCR Pipeline** designed to:
- reduce search space
- prevent crashes
- automatically select the best OCR configuration


## System Architecture

### Phase A — Orientation & Baseline Search

- Iterates over:
  - image rotations (0°, 90°, 180°, 270°)
  - Tesseract PSM modes
  - language configurations (rus, eng, rus+eng)
- Uses fast OCR (`image_to_string`)
- Determines the best orientation and baseline configuration
- Applies early stopping to limit computation

### Phase B — Focused Refinement

- Runs only on the best orientation selected in Phase A
- Uses `image_to_data` for detailed OCR analysis
- Selects the final result using a composite heuristic score:
  - text length
  - average confidence
  - noise and garbage penalties

This approach:
- significantly reduces OCR calls
- avoids Tesseract memory overflows
- improves overall pipeline stability


## Image Preprocessing

Multiple preprocessing variants are evaluated automatically:

- grayscale / inverted grayscale
- adaptive thresholding
- sharpening
- bilateral filtering

Each variant is scored independently, and the best result is selected dynamically.


## Text Post-processing

### General Normalization
- removal of garbage characters
- whitespace normalization
- correction of common OCR artifacts

### Caption-Oriented Cleanup
A specialized sanitizer is applied **only for caption-like text**:
- fixes merged or split words
- restores missing punctuation
- removes outline-related noise common in meme images


## Stability & Error Handling

Explicit safeguards include:
- limited use of `image_to_data`
- candidate pre-selection
- downscaling of large images
- graceful fallback on Tesseract errors
- no unbounded brute-force search

As a result, the pipeline remains stable even on complex or noisy images.


## How to Run

### Requirements
- Python 3.9+
- Tesseract OCR (installed and available in PATH)

Install dependencies:
```bash
pip install -r requirements.txt
````

Run:

```bash
python src/main.py path/to/image.jpg
```


## Example Output

```
=== RVisionT Universal OCR (Two-Phase Stable) ===
Text:
It was the best of times, it was the worst of times,
it was the age of wisdom, it was the age of foolishness...

Confidence: 95.9
```

## Known Limitations

* Heuristic-based scoring (no labeled dataset)
* Limited language support (Russian / English)
* OCR speed may be slow on large images
* ROI detection is heuristic, not layout-aware

## Educational Value

This project demonstrates:

* practical Computer Vision techniques
* engineering strategies for unstable libraries
* search-space optimization
* noisy data handling
* systematic pipeline design

It is suitable as:

* a research or coursework project
* a foundation for future OCR extensions
* a portfolio example of engineering thinking

## Author

**XCON | RX**
TG: [@End1essspace](https://t.me/End1essspace)
GitHub: [End1essspace](https://github.com/End1essspace)

Developed for educational and research purposes.
All logic is implemented manually without using ready-made end-to-end OCR frameworks.
