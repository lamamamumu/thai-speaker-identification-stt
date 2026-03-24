"""
evaluate.py — WER and DER evaluation functions.

WER: compare full STT transcript vs full reference text (joined).
DER: sliding-window RF diarization vs label_sound.xlsx reference.
     Completely independent of STT segmentation boundaries.
"""

import os
import numpy as np
import pandas as pd
from jiwer import wer


def compute_wer(ref_df, hyp_segs, label):
    """
    Character-level WER for Thai text.

    Thai has no spaces between words — Google STT adds spaces between every
    word token ('มนุษย์ ทุก คน') but reference labels have natural Thai
    ('มนุษย์ทุกคน'). Word-level WER treats these as completely different,
    inflating the score massively.

    Fix: strip ALL spaces from both reference and hypothesis, then compute
    WER at the character level (each character = one 'word' token).
    This gives a meaningful accuracy score regardless of tokenisation.
    """
    ref_text = " ".join(str(v) for v in ref_df['Content'] if str(v).strip())
    hyp_text = " ".join(s['text'] for s in hyp_segs if s.get('text', '').strip())

    # Strip spaces → character-level comparison
    ref_chars = list(ref_text.replace(" ", ""))
    hyp_chars = list(hyp_text.replace(" ", ""))

    ref_c = len(ref_chars)
    hyp_c = len(hyp_chars)
    print(f"    {label:<8}: ref={ref_c:>6} chars, hyp={hyp_c:>6} chars", end="")

    if ref_chars and hyp_chars:
        # jiwer expects space-separated tokens — join chars with spaces
        score = wer(" ".join(ref_chars), " ".join(hyp_chars))
        print(f" → CER = {score:.2%}")
        return score
    print(" → WER skipped (empty)")
    return None


def print_wer_table(wer_g_train, wer_w_train, wer_g_test, wer_w_test,
                    wer_g_all, wer_w_all):
    def f(v):
        return f"{v:.2%}" if v is not None else "N/A"

    print(f"\n  {'Engine':<10} {'Train CER':>10}  {'Test CER':>10}  {'Combined CER':>13}")
    print("  " + "─" * 50)
    for engine, wtr, wte, wco in [
        ("Google",  wer_g_train, wer_g_test, wer_g_all),
        ("Whisper", wer_w_train, wer_w_test, wer_w_all),
    ]:
        print(f"  {engine:<10} {f(wtr):>10}  {f(wte):>10}  {f(wco):>13}")
    print()
    print("  ℹ️  CER = Character Error Rate (spaces stripped before comparison).")
    print("  ℹ️  Thai has no natural word boundaries so CER is more meaningful than WER.")
    print("  ℹ️  Use DER as the primary evaluation metric for this dataset.")


def save_wer_txt(path, wer_g_test, wer_w_test, df_cnt_test, g_test, w_test):
    """Save test WER detail (reference vs both hypotheses) to a .txt file."""
    def f(v):
        return f"{v:.2%}" if v is not None else "N/A"

    ref_text = " ".join(
        str(v) for v in df_cnt_test['Content'] if str(v).strip()
    ) if not df_cnt_test.empty else ""
    g_text = " ".join(s['text'] for s in g_test if s.get('text', '').strip())
    w_text = " ".join(s['text'] for s in w_test if s.get('text', '').strip())

    lines = [
        "CER — Test Set Detail (Character Error Rate)",
        "=" * 60, "",
        "Engine          CER",
        f"Google          {f(wer_g_test)}",
        f"Whisper         {f(wer_w_test)}",
        "",
        "=" * 60,
        "REFERENCE (label_speech.xlsx — test rows joined)",
        "=" * 60,
        ref_text or "(no reference labels in test window)",
        "",
        "=" * 60,
        f"HYPOTHESIS — Google STT ({len(g_test)} segments)",
        "=" * 60,
        g_text or "(no segments)",
        "",
        "=" * 60,
        f"HYPOTHESIS — Whisper STT ({len(w_test)} segments)",
        "=" * 60,
        w_text or "(no segments)",
    ]
    with open(path, 'w', encoding='utf-8-sig') as fp:
        fp.write("\n".join(lines))
    print(f"\n   💾 Test WER detail saved → {path}")


def sliding_window_diarize(y_audio, processor, classifier, sr,
                            start_s, end_s, win_s=0.5, step_s=0.25,
                            smooth_k=5):
    """
    Slide a fixed window across audio, predict speaker for each window.
    Apply majority-vote smoothing over smooth_k consecutive windows
    to reduce single-window prediction noise, then merge same-speaker runs.

    Returns list of {'start', 'end', 'speaker'}.
    """
    from collections import Counter

    raw = []
    t = start_s
    while t + win_s <= end_s:
        s_idx = int((t - start_s) * sr)
        e_idx = int((t - start_s + win_s) * sr)
        y_win = y_audio[s_idx:e_idx]
        feat  = processor.extract_mfcc(y_win, sr)
        spk   = classifier.predict(feat) if (feat is not None and len(feat) == 160) else "Unknown"
        raw.append({'start': round(t, 3), 'end': round(t + win_s, 3), 'speaker': spk})
        t += step_s

    if not raw:
        return []

    # ── Majority-vote smoothing ───────────────────────────────────────
    # For each window i, take the majority speaker in [i-k, i+k]
    half = smooth_k // 2
    smoothed = []
    for i, seg in enumerate(raw):
        lo  = max(0, i - half)
        hi  = min(len(raw), i + half + 1)
        nbr = [raw[j]['speaker'] for j in range(lo, hi)]
        winner = Counter(nbr).most_common(1)[0][0]
        smoothed.append({**seg, 'speaker': winner})

    # ── Merge consecutive same-speaker windows ────────────────────────
    merged = []
    for seg in smoothed:
        if (merged and merged[-1]['speaker'] == seg['speaker']
                and abs(merged[-1]['end'] - seg['start']) < step_s + 0.01):
            merged[-1]['end'] = seg['end']
        else:
            merged.append(dict(seg))
    return merged


def assign_spk_from_hyp(stt_segs, hyp_segs):
    """
    Map each STT segment to the speaker with the most time overlap
    in the sliding-window diarization hypothesis.
    """
    out = []
    for seg in stt_segs:
        best_spk, best_overlap = "Unknown", 0.0
        for h in hyp_segs:
            ov = min(seg['end'], h['end']) - max(seg['start'], h['start'])
            if ov > best_overlap:
                best_overlap = ov
                best_spk = h['speaker']
        out.append({**seg, 'speaker': best_spk})
    return out