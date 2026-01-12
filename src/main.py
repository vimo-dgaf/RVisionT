import os
import sys
import json
import time
import re
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pytesseract


@dataclass
class OcrResult:
    text: str
    confidence: float  # mean conf if available else -1
    meta: Dict[str, object]


# ----------------------------
# Normalization helpers
# ----------------------------
_LAT2CYR = str.maketrans({
    "A": "А", "a": "а",
    "B": "В", "b": "в",
    "C": "С", "c": "с",
    "E": "Е", "e": "е",
    "H": "Н", "h": "н",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "T": "Т", "t": "т",
    "X": "Х", "x": "х",
    "Y": "У", "y": "у",
})


def _latin_to_cyr_lookalikes(s: str) -> str:
    return s.translate(_LAT2CYR)


def _count_cyr(s: str) -> int:
    return sum(("А" <= ch <= "я") or (ch in "Ёё") for ch in s)


def _count_lat(s: str) -> int:
    return sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in s)


def _script_ratio_cyr(s: str) -> float:
    cyr = _count_cyr(s)
    lat = _count_lat(s)
    tot = cyr + lat
    return (cyr / tot) if tot else 0.0


def _postprocess_tokens_for_captions(s: str) -> str:
    """
    Small, safe heuristics to remove typical OCR garbage:
    - single-letter token between ALL-CAPS words ("... НАЗЫВАЕТСЯ а МЕМЫ ...")
    - trailing short junk like "ПУ)" / "Tn)".
    """
    if not s:
        return s

    tokens = s.split()
    if len(tokens) < 3:
        return s

    def _caps_word(t: str) -> bool:
        letters = [ch for ch in t if ch.isalpha() or ch in "Ёё"]
        return len(letters) >= 4 and all(ch.isupper() for ch in letters)

    cleaned = []
    for i, tok in enumerate(tokens):
        if len(tok) == 1 and tok.isalpha():
            prev_tok = tokens[i - 1] if i - 1 >= 0 else ""
            next_tok = tokens[i + 1] if i + 1 < len(tokens) else ""
            if _caps_word(prev_tok) and _caps_word(next_tok):
                continue
        cleaned.append(tok)
    tokens = cleaned

    while tokens:
        last = tokens[-1]
        letters = sum((ch.isalpha() or ch in "Ёё") for ch in last)
        has_bracket = any(ch in ")]}" for ch in last)
        if len(last) <= 4 and letters <= 2 and has_bracket:
            tokens.pop()
            continue
        break

    return " ".join(tokens)




def _postprocess_rus_common(s: str) -> str:
    """Safe fixes for frequent Russian OCR errors in captions/UI."""
    if not s:
        return s
    t = s

    # normalize quotes/apostrophes used as artifacts
    t = t.replace("’", "'").replace("`", "'").replace("«", '"').replace("»", '"')

    # fix missing spaces around НЕ in common patterns
    t = re.sub(r"\bНЕУБИРАЕШЬСЯ\b", "НЕ УБИРАЕШЬСЯ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bНЕМОЕШЬ\b", "НЕ МОЕШЬ", t, flags=re.IGNORECASE)
    t = re.sub(r"\bНЕУМЕЕШЬ\b", "НЕ УМЕЕШЬ", t, flags=re.IGNORECASE)

    # ДОМАОНЕ / ДОМАЭНЕ / ДОМАЫНЕ -> ДОМА? НЕ
    t = re.sub(r"\bДОМА[ОЭЫAЗ]?НЕ\b", "ДОМА? НЕ", t, flags=re.IGNORECASE)

    # remove stray braces and pipes often produced near edges
    t = re.sub(r"[{}|]+", " ", t)

    # collapse spaces
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t

def _sanitize_meme_ru(s: str) -> str:
    """Aggressive but safe cleanup for RU meme captions (applied only if caption-like + Cyrillic-dominant)."""
    if not s:
        return s
    t = s

    # remove quote-like artifacts
    t = t.replace("’", "'").replace("`", "'").replace("«", '"').replace("»", '"')
    t = re.sub(r"[\"'`]+", "", t)

    # common stroke artifacts near words
    t = re.sub(r"ПОСУДУЭ", "ПОСУДУ", t, flags=re.IGNORECASE)
    t = re.sub(r"ПОСУДУ[ЭЕ]?\s*[:;]?", "ПОСУДУ", t, flags=re.IGNORECASE)

    # "НЕУМЕЕШЬЯ" -> "НЕ УМЕЕШЬ?"
    t = re.sub(r"\bНЕ\s*УМЕЕШЬЯ\b", "НЕ УМЕЕШЬ?", t, flags=re.IGNORECASE)
    t = re.sub(r"\bНЕУМЕЕШЬЯ\b", "НЕ УМЕЕШЬ?", t, flags=re.IGNORECASE)

    # ensure space after НЕ (if glued)
    t = re.sub(r"\bНЕ([А-ЯЁ])", r"НЕ \1", t)

    # normalize "готовить" tail to question
    t = re.sub(r"\bготовить[эе]?\b", "ГОТОВИТЬ?", t, flags=re.IGNORECASE)
    t = re.sub(r"\bгото[ао]ритее\b", "ГОТОВИТЬ?", t, flags=re.IGNORECASE)

    # cut off junk after final question word
    t = re.sub(r"(ГОТОВИТЬ\?)(.*)$", r"\1", t, flags=re.IGNORECASE)

    # remove braces/pipes
    t = re.sub(r"[{}|]+", " ", t)

    # final spacing/punct cleanup
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def _looks_like_caption(text: str) -> bool:
    if not text:
        return False
    # lots of uppercase or typical meme punctuation
    letters = [ch for ch in text if ch.isalpha() or ch in "Ёё"]
    if not letters:
        return False
    upper = sum(ch.isupper() for ch in letters)
    up_ratio = upper / max(1, len(letters))
    return up_ratio > 0.55 or "!" in text or "?" in text
def _cleanup_text(s: str) -> str:
    """
    Conservative cleanup: keep alnum + basic punctuation, normalize spaces.
    IMPORTANT: apply latin->cyr lookalikes ONLY when text is mostly Cyrillic.
    """
    if not s:
        return ""

    out = []
    for ch in s:
        if ch.isalnum() or ch in " .,:;!?-()[]{}\"'/%+" or ch in "Ёё":
            out.append(ch)
        else:
            out.append(" ")
    s = "".join(out)
    s = " ".join(s.split())

    # Apply lookalike mapping only for mostly-Cyrillic strings
    if _script_ratio_cyr(s) >= 0.55:
        s = _latin_to_cyr_lookalikes(s)

    tokens = s.split()
    fixed = []
    for t in tokens:
        if any(ch.isalpha() or ch in "Ёё" for ch in t):
            t = re.sub(r"(\D)\d+$", r"\1", t)
        fixed.append(t)

    # remove immediate duplicates
    dedup = []
    for w in fixed:
        if not dedup or dedup[-1] != w:
            dedup.append(w)

    s = " ".join(dedup)
    s = s.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)

    return s.strip()


# ----------------------------
# Scoring (avoid long garbage winning)
# ----------------------------
_ALLOWED_CHARS = set(" Ёё.,!?-:;()[]{}\"'/%+")


def _garbage_penalty(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 1e9

    total = max(1, len(t))
    letters = sum((ch.isalpha() or ch in "Ёё") for ch in t)
    spaces = t.count(" ")

    bad = 0
    for ch in t:
        if ch.isalnum() or ch in _ALLOWED_CHARS:
            continue
        bad += 1
    bad_ratio = bad / total
    space_ratio = spaces / total

    cyr = _count_cyr(t)
    lat = _count_lat(t)
    mix_ratio = (min(cyr, lat) / max(1, (cyr + lat))) if (cyr + lat) else 0.0

    penalty = 0.0
    if letters < 6:
        penalty += 800.0
    penalty += bad_ratio * 2500.0
    if len(t) > 28 and space_ratio < 0.03:
        penalty += 650.0
    penalty += mix_ratio * 1400.0

    if len(t) > 200:
        penalty += (len(t) - 200) * 4.0

    singles = sum(1 for w in t.split() if len(w) == 1)
    if singles >= 6:
        penalty += singles * 60.0

    return penalty


def _quality_score(text: str, conf: float) -> float:
    t = (text or "").strip()
    if not t:
        return -1e18

    cyr = _count_cyr(t)
    lat = _count_lat(t)
    letters = sum((ch.isalpha() or ch in "Ёё") for ch in t)
    digits = sum(ch.isdigit() for ch in t)
    punct = sum((not ch.isalnum()) and (ch not in "Ёё") for ch in t)

    words = t.split()
    long_words = sum(1 for w in words if len(w) >= 5)
    short_words = sum(1 for w in words if len(w) <= 2)

    score = 0.0
    score += letters * 6.0
    score += long_words * 18.0
    score -= short_words * 10.0
    score -= punct * 1.0
    score -= digits * 0.6

    if cyr > 0 and lat == 0:
        score += cyr * 8.0
    elif lat > 0 and cyr == 0:
        score += lat * 5.0
    else:
        score -= min(cyr, lat) * 10.0

    if conf >= 0:
        score += conf * 2.2
        if conf < 55:
            score -= (55 - conf) * 35.0

    score += min(len(t), 160) * 0.05
    score -= _garbage_penalty(t)
    return score


# ----------------------------
# Simple text-band detection (ROI segmentation)
# ----------------------------
def _detect_text_bands(gray: np.ndarray, max_bands: int = 4) -> List[Tuple[int, int]]:
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return [(0, h)]

    target_w = 900
    if w > target_w:
        scale = target_w / float(w)
        g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        g = gray.copy()

    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    g_blur = cv2.GaussianBlur(g, (3, 3), 0)

    _, th = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)

    def _row_density(bin_img: np.ndarray) -> np.ndarray:
        fg = (bin_img > 0).astype(np.uint8)
        dens = fg.sum(axis=1).astype(np.float32) / float(fg.shape[1])
        k = max(5, int(round(fg.shape[0] * 0.015)))
        if k % 2 == 0:
            k += 1
        dens = cv2.GaussianBlur(dens.reshape(-1, 1), (1, k), 0).reshape(-1)
        return dens

    d1 = _row_density(th)
    d2 = _row_density(th_inv)
    dens = d1 if float(d1.mean()) < float(d2.mean()) else d2

    thr = max(0.012, float(np.percentile(dens, 78)))
    mask = dens > thr

    bands_small: List[Tuple[int, int]] = []
    y = 0
    min_run = max(8, int(round(g.shape[0] * 0.02)))
    while y < len(mask):
        if not mask[y]:
            y += 1
            continue
        y0 = y
        while y < len(mask) and mask[y]:
            y += 1
        y1 = y
        if (y1 - y0) >= min_run:
            bands_small.append((y0, y1))

    if not bands_small:
        return [(0, h)]

    bands_small.sort()
    merged: List[Tuple[int, int]] = []
    gap = max(6, int(round(g.shape[0] * 0.015)))
    cur0, cur1 = bands_small[0]
    for a, b in bands_small[1:]:
        if a - cur1 <= gap:
            cur1 = max(cur1, b)
        else:
            merged.append((cur0, cur1))
            cur0, cur1 = a, b
    merged.append((cur0, cur1))

    scored = []
    for y0, y1 in merged:
        score = (y1 - y0) * float(dens[y0:y1].mean())
        scored.append((score, y0, y1))
    scored.sort(reverse=True)

    picked = scored[:max_bands]
    picked.sort(key=lambda x: x[1])

    scale_back = float(h) / float(g.shape[0])
    pad = int(round(18 * scale_back))
    out: List[Tuple[int, int]] = []
    for _, y0s, y1s in picked:
        y0 = int(round(y0s * scale_back)) - pad
        y1 = int(round(y1s * scale_back)) + pad
        y0 = max(0, y0)
        y1 = min(h, y1)
        if y1 - y0 >= 18:
            out.append((y0, y1))

    out.sort()
    final: List[Tuple[int, int]] = []
    for y0, y1 in out:
        if not final:
            final.append((y0, y1))
            continue
        p0, p1 = final[-1]
        if y0 <= p1 + 10:
            final[-1] = (p0, max(p1, y1))
        else:
            final.append((y0, y1))

    # Meme split heuristic: if one broad band covers most of the image, try top/bottom
    if len(final) == 1:
        y0, y1 = final[0]
        if (y1 - y0) > int(0.75 * h):
            t0, t1 = 0, int(round(0.28 * h))
            b0, b1 = int(round(0.62 * h)), h
            return [(t0, t1), (b0, b1)]

    return final if final else [(0, h)]


# ----------------------------
# Dedup merge
# ----------------------------
def _norm_line_for_containment(x: str) -> str:
    x = _cleanup_text(x)
    return " ".join(x.lower().split())


def _merge_texts(bestA: str, bestB: str, metaA: Dict[str, object], metaB: Dict[str, object]) -> str:
    a = (bestA or "").strip()
    b = (bestB or "").strip()
    if not a:
        return b
    if not b:
        return a

    a_n = _norm_line_for_containment(a)
    b_n = _norm_line_for_containment(b)

    if a_n and a_n in b_n and len(b_n) > len(a_n):
        return b
    if b_n and b_n in a_n and len(a_n) > len(b_n):
        return a

    def _meta_score(m: Dict[str, object]) -> float:
        try:
            return float(m.get("score", "-1e18"))
        except Exception:
            return -1e18

    scoreA = _meta_score(metaA) + _quality_score(a, -1.0)
    scoreB = _meta_score(metaB) + _quality_score(b, -1.0)
    return a if scoreA >= scoreB else b


# ----------------------------
# Preprocessing variants
# ----------------------------
def _unsharp(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)


def _prep_variants(gray0: np.ndarray, phase: str) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = []
    gray0 = cv2.normalize(gray0, None, 0, 255, cv2.NORM_MINMAX)

    scales = (2.0, 3.0) if phase == "A" else (3.0,)

    for scale in scales:
        gray = cv2.resize(gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        variants.append((f"gray_s{scale}", gray))
        variants.append((f"inv_s{scale}", cv2.bitwise_not(gray)))

        sharp = _unsharp(gray)
        variants.append((f"sharp_s{scale}", sharp))
        variants.append((f"sharp_inv_s{scale}", cv2.bitwise_not(sharp)))

        if phase == "B":
            th = cv2.adaptiveThreshold(
                sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
            )
            th_inv = cv2.bitwise_not(th)
            variants.append((f"th_s{scale}", th))
            variants.append((f"th_inv_s{scale}", th_inv))

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            close = cv2.morphologyEx(th_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
            variants.append((f"close_s{scale}", close))

            dil = cv2.dilate(close, kernel, iterations=1)
            variants.append((f"dil_s{scale}", dil))

    den = cv2.bilateralFilter(gray0, 7, 50, 50)
    variants.append(("bilateral", den))
    return variants


# ----------------------------
# Contrast + deskew
# ----------------------------
def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _rotate_bound(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    (h, w) = gray.shape[:2]
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(gray, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    if h < 60 or w < 60:
        return gray, 0.0

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    use = th_inv if th_inv.mean() < th.mean() else th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    use = cv2.morphologyEx(use, cv2.MORPH_CLOSE, k, iterations=1)

    coords = cv2.findNonZero(use)
    if coords is None or len(coords) < 250:
        return gray, 0.0

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 1.0 or abs(angle) > 10.0:
        return gray, 0.0

    return _rotate_bound(gray, angle), float(angle)


# ----------------------------
# OCR search
# ----------------------------

def _run_search(
    rotations: List[Tuple[str, np.ndarray]],
    psms: List[int],
    lang_candidates: List[str],
    phase: str,
    progress_every: int,
    early_stop: bool,
    early_stop_score: float,
    verbose: bool = True,
) -> Tuple[OcrResult, Optional[np.ndarray]]:

    """
    Robust OCR search.
    Key stability choices:
      - Prefer image_to_string for most candidates (cheaper, less memory).
      - Call image_to_data only for top candidates to estimate confidence.
      - Skip tiny ROIs and downscale very large images to avoid tesseract std::bad_alloc.
      - Catch TesseractError and continue (do not crash the pipeline).
    """
    best = OcrResult(text="", confidence=-1.0, meta={"reason": f"init_phase_{phase}"})
    best_score = -1e18
    best_preview: Optional[np.ndarray] = None

    t0 = time.time()
    done = 0
    total = 0

    for _, gray in rotations:
        variants = _prep_variants(gray, phase=phase)
        total += len(variants) * len(psms) * len(lang_candidates)

    def _maybe_downscale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        max_pixels = 5_000_000  # ~5MP
        if h * w <= max_pixels:
            return img
        scale = math.sqrt(max_pixels / float(h * w))
        nw = max(64, int(w * scale))
        nh = max(64, int(h * scale))
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    def _safe_to_ocr(img: np.ndarray) -> bool:
        h, w = img.shape[:2]
        if h < 40 or w < 120:
            return False
        if h * w < 40_000:
            return False
        return True

    for rot_name, gray in rotations:
        variants = _prep_variants(gray, phase=phase)

        for var_name, img0 in variants:
            img = _maybe_downscale(img0)
            if not _safe_to_ocr(img):
                continue

            for psm in psms:
                cfg = f'--oem 3 --psm {psm} -c user_defined_dpi=300 -c preserve_interword_spaces=1'

                for lang in lang_candidates:
                    done += 1
                    if progress_every > 0 and done % progress_every == 0:
                        elapsed = time.time() - t0
                        avg = elapsed / max(1, done)
                        eta = max(0.0, (total - done) * avg)
                        if verbose:
                            print(
                                f"[phase {phase}] [{done}/{total}] elapsed={elapsed:.1f}s avg={avg:.3f}s ETA~{eta:.1f}s best_score={best_score:.2f}"
                            )

                    # 1) fast path: text only
                    try:
                        raw = pytesseract.image_to_string(img, lang=lang, config=cfg, timeout=12)
                    except pytesseract.TesseractError as e:
                        # tesseract internal failure (e.g. std::bad_alloc) -> skip candidate
                        continue
                    except Exception:
                        continue

                    cleaned = _cleanup_text(raw)
                    if not cleaned:
                        continue

                    # Score without confidence first
                    score0 = _quality_score(cleaned, -1.0)

                    # Only compute confidence for contenders (expensive / riskier)
                    avg_conf = -1.0
                    score = score0
                    if score0 > (best_score - 40.0):
                        try:
                            data = pytesseract.image_to_data(
                                img, lang=lang, config=cfg, output_type=pytesseract.Output.DICT, timeout=12
                            )

                            confs = []
                            # data['conf'] can be strings; -1 means non-text blocks
                            for conf in data.get("conf", []):
                                try:
                                    c = float(conf)
                                except Exception:
                                    continue
                                if c >= 0:
                                    confs.append(c)

                            avg_conf = float(np.mean(confs)) if confs else -1.0
                            score = _quality_score(cleaned, avg_conf)
                        except pytesseract.TesseractError:
                            # keep score0, conf=-1
                            avg_conf = -1.0
                            score = score0
                        except Exception:
                            avg_conf = -1.0
                            score = score0

                    if score > best_score:
                        best_score = score
                        best = OcrResult(
                            text=cleaned,
                            confidence=avg_conf,
                            meta={
                                "phase": phase,
                                "rotation": rot_name,
                                "variant": var_name,
                                "psm": str(psm),
                                "lang": lang,
                                "score": f"{score:.2f}",
                            },
                        )
                        best_preview = img
                        if verbose:
                            print(
                                f"NEW BEST [phase {phase}] score={score:.2f} lang={lang} psm={psm} rot={rot_name} var={var_name} conf={avg_conf:.1f} preview={cleaned[:80]!r}"
                            )

                        if early_stop and best_score >= early_stop_score:
                            words = best.text.split()
                            cyr = _count_cyr(best.text)
                            lat = _count_lat(best.text)

                            ok = False
                            if best.confidence >= 70 and len(words) >= 3:
                                # Russian-looking
                                if cyr >= 12 and lat <= 3:
                                    ok = True
                                # English-looking
                                if lat >= 18 and cyr <= 3:
                                    ok = True

                            if ok:
                                print(f"EARLY STOP [phase {phase}]: threshold reached.")
                                return best, best_preview

    return best, best_preview


# ----------------------------
# Universal extraction
# ----------------------------
def extract_text_universal(image_path: str, out_dir: str = "output", progress_every: int = 50) -> OcrResult:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    bgr = cv2.copyMakeBorder(bgr, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray0 = _apply_clahe(gray0)
    gray0, deskew_angle = _deskew(gray0)

    rotations = [
        ("rot0", gray0),
        ("rot90", cv2.rotate(gray0, cv2.ROTATE_90_CLOCKWISE)),
        ("rot270", cv2.rotate(gray0, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    lang_candidates = ["rus+eng", "eng", "rus"]

    # Phase A: orientation must be SAFE (no aggressive threshold variants)
    bestA, prevA = _run_search(
        rotations=rotations,
        psms=[3, 4, 6],
        lang_candidates=lang_candidates,
        phase="A",
        progress_every=progress_every,
        early_stop=True,
        early_stop_score=520.0,
    )

    rotA = str(bestA.meta.get("rotation", "rot0"))
    rot_map = {name: img for name, img in rotations}
    grayA = rot_map.get(rotA, gray0)

    bestB, prevB = _run_search(
        rotations=[(rotA, grayA)],
        psms=[11, 6, 7, 8],
        lang_candidates=lang_candidates,
        phase="B",
        progress_every=progress_every,
        early_stop=True,
        early_stop_score=560.0,
    )

    # ROI fallback when full-frame is weak OR when layout strongly suggests captions (2+ bands)
    bands = _detect_text_bands(grayA, max_bands=4)

    hA = grayA.shape[0]
    min_h = int(round(hA * 0.06))
    spans_far = (bands[-1][0] - bands[0][1]) >= int(round(0.18 * hA)) if len(bands) >= 2 else False
    covers_top_bottom = (bands[0][0] <= int(round(0.22 * hA)) and bands[-1][1] >= int(round(0.70 * hA))) if len(bands) >= 2 else False
    roi_layout_ok = (
        len(bands) >= 2
        and all((y1 - y0) >= min_h for (y0, y1) in bands)
        and (spans_far or covers_top_bottom)
    )

    full_conf_weak = (bestB.confidence >= 0 and bestB.confidence < 75)
    full_text_short = (len(bestB.text or "") < 22)
    # If layout has captions, prefer ROI even if full-frame is "ok-ish"
    try_roi = roi_layout_ok and (full_conf_weak or full_text_short or len(bands) >= 2)

    if try_roi:
        roi_texts: List[str] = []
        roi_info: List[Dict[str, object]] = []
        best_conf = bestB.confidence
        best_preview: Optional[np.ndarray] = prevB

        for (y0, y1) in bands:
            crop = grayA[y0:y1, :]
            best_i, prev_i = _run_search(
                rotations=[(rotA, crop)],
                psms=[7, 6, 11],
                lang_candidates=lang_candidates,
                phase="B",
                progress_every=0,
                early_stop=True,
                early_stop_score=540.0,
            )
            roi_info.append({"band": [int(y0), int(y1)], "best": best_i.meta})
            if best_i.text:
                roi_texts.append(best_i.text)
            if best_i.confidence >= 0:
                best_conf = max(best_conf, best_i.confidence)
            if best_preview is None and prev_i is not None:
                best_preview = prev_i

        roi_text = _cleanup_text("\n".join([t for t in roi_texts if t]).strip())
        if roi_text:
            roi_score = _quality_score(roi_text, best_conf)
            try:
                full_score = float(bestB.meta.get("score", "-1e18"))
            except Exception:
                full_score = _quality_score(bestB.text or "", bestB.confidence)

            if roi_score > full_score + 10.0:
                bestB = OcrResult(
                    text=roi_text,
                    confidence=best_conf,
                    meta={
                        "phase": "B_ROI",
                        "rotation": rotA,
                        "bands": roi_info,
                        "score": f"{roi_score:.2f}",
                        "note": "ROI fallback: OCR per detected text band, concat top-to-bottom",
                    },
                )
                prevB = best_preview

    final_text = _cleanup_text(_merge_texts(bestA.text, bestB.text, bestA.meta, bestB.meta))
    # Caption-oriented cleanup (does not affect regular documents much)
    if _looks_like_caption(final_text):
        # caption cleanup only at the very end
        final_text = _postprocess_tokens_for_captions(final_text)

        if _script_ratio_cyr(final_text) >= 0.55:
            final_text = _postprocess_rus_common(final_text)
            final_text = _sanitize_meme_ru(final_text)

    conf = bestB.confidence if (bestB.confidence >= 0 and bestB.confidence >= bestA.confidence) else bestA.confidence

    combined = OcrResult(
        text=final_text,
        confidence=conf,
        meta={
            "bestA": bestA.meta,
            "bestB": bestB.meta,
            "deskew_angle": deskew_angle,
            "note": "stable two-phase: Phase A chooses orientation; Phase B runs only on that rotation",
        }
    )

    base = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(out_dir, f"{base}_text.txt")
    json_path = os.path.join(out_dir, f"{base}_result.json")
    prevA_path = os.path.join(out_dir, f"{base}_bestA.png")
    prevB_path = os.path.join(out_dir, f"{base}_bestB.png")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(combined.text + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"image": image_path, "text": combined.text, "confidence": combined.confidence, "meta": combined.meta},
            f,
            ensure_ascii=False,
            indent=2
        )

    if prevA is not None:
        cv2.imwrite(prevA_path, prevA)
    if prevB is not None:
        cv2.imwrite(prevB_path, prevB)

    return combined


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path: ").strip().strip('"')

    result = extract_text_universal(image_path=image_path, out_dir="output", progress_every=50)
    print("\n=== RVisionT Universal OCR (Two-Phase Stable) ===")
    print("Text:")
    print(result.text if result.text else "[EMPTY]")
    print("\nMeta:", result.meta)
    print("Confidence:", result.confidence)


if __name__ == "__main__":
    main()
