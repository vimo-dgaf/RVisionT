# RVisionT — Known Limitations

This document lists the known limitations and non-goals of the RVisionT project.
They are documented intentionally to clarify scope, design trade-offs, and
current boundaries of the system.

RVisionT is an **engineering and research project**, not a production OCR system.


## 1. Heuristic-Based Scoring

Text quality evaluation relies entirely on handcrafted heuristics:

* character composition
* word length and structure
* script consistency (Cyrillic vs Latin)
* confidence weighting (when available)

There is **no labeled dataset** and no learned scoring model.
As a result, scoring may fail on edge cases or unconventional text layouts.


## 2. Limited Language Support

Currently supported languages:

* Russian (`rus`)
* English (`eng`)
* Russian + English (`rus+eng`)

Other languages, scripts, and right-to-left text are **not supported**.


## 3. Performance Constraints

RVisionT performs multiple OCR passes to improve stability and result quality.

Consequences:

* slower execution on large images
* high CPU usage during processing
* not suitable for real-time OCR scenarios

No parallel processing or caching is implemented.


## 4. Heuristic ROI Detection

Text region (ROI) detection is based on simple image heuristics:

* horizontal text density
* band merging rules
* caption-oriented assumptions

This approach may fail on:

* complex multi-column layouts
* tables or forms
* dense full-page documents

RVisionT does **not** perform full document layout analysis.


## 5. OCR Engine Dependency

RVisionT depends on Tesseract OCR as an external engine.

Known implications:

* behavior varies across Tesseract versions
* results depend on language pack quality
* rare engine-level crashes may still occur

RVisionT mitigates these issues but cannot fully eliminate them.


## 6. Orientation Handling Limitations

Orientation detection is limited to:

* 0°
* 90°
* 270°

Images rotated by 180° may produce suboptimal results.


## 7. No Production Interface

RVisionT does not provide:

* REST or gRPC APIs
* batch-processing interfaces
* GUI applications
* packaging as a pip-installable library

These are considered out of scope for the current project.


## 8. No Formal Benchmarking

There is no:

* CER / WER evaluation
* standardized dataset comparison
* performance benchmarking suite

Evaluation is qualitative and example-driven.


## Summary

These limitations are **intentional design trade-offs**, not oversights.

RVisionT prioritizes:

* robustness
* stability
* explicit failure handling
* architectural clarity

over production readiness or maximum OCR accuracy.
