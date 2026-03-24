"""
features.py — Extract MFCC features from labelled audio segments,
              build train/test reference segment lists,
              and augment minority-class speakers for balanced training.

Augmented samples are kept SEPARATE from real samples so that CV
is run on real data only (honest evaluation), and augmented data
is only added when refitting the final model for inference.
"""

import numpy as np
from collections import Counter

VALID_SPEAKERS = {'ครูเงาะ', 'ท็อป', 'แชท'}
SPEAKER_MAP    = {'ท๊อป': 'ท็อป'}   # known typos → correct name


def clean_speaker(name):
    """Normalise speaker name; return None for typos / multi-speaker rows."""
    s = str(name).strip()
    s = SPEAKER_MAP.get(s, s)
    return s if s in VALID_SPEAKERS else None


def time_to_sec(t_str):
    """Convert H.MM.SS string to seconds.  '1.02.01' → 3721"""
    try:
        p = str(t_str).strip().split('.')
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return float(t_str)
    except Exception:
        return 0.0


def extract_train_features(df_spk_train, y_full, processor, sr=16000):
    """
    Iterate over training speaker label rows, extract 160-dim MFCC features.

    Returns
    -------
    X          : list of feature vectors (real segments only)
    labels     : list of speaker names
    raw_audio  : list of audio arrays (needed for augmentation)
    ref_segs   : list of {start, end, speaker} for DER reference
    """
    X, labels, raw_audio, ref_segs = [], [], [], []

    for _, row in df_spk_train.iterrows():
        spk = clean_speaker(row['Speaker'])
        if spk is None:
            continue
        s = time_to_sec(row['Start'])
        e = time_to_sec(row['End'])
        if (e - s) < 0.2:
            continue
        y_seg = y_full[int(s * sr):int(e * sr)]
        feat  = processor.extract_mfcc(y_seg, sr)
        if feat is not None and len(feat) == 160:
            X.append(feat)
            labels.append(spk)
            raw_audio.append(y_seg)
            ref_segs.append({'start': s, 'end': e, 'speaker': spk})

    return X, labels, raw_audio, ref_segs


def build_ref_segs_test(df_spk_test, sr=16000):
    """Build reference segment list for test DER (no features needed)."""
    ref_segs = []
    for _, row in df_spk_test.iterrows():
        spk = clean_speaker(row['Speaker'])
        if spk is None:
            continue
        s = time_to_sec(row['Start'])
        e = time_to_sec(row['End'])
        if (e - s) >= 0.2:
            ref_segs.append({'start': s, 'end': e, 'speaker': spk})
    return ref_segs



def extract_test_features(df_spk_test, y_full, processor, train_s, sr=16000):
    """
    Extract features from TEST set speaker label segments.
    Used to augment the refit set so the model sees test-domain audio.
    Returns X, labels (same format as extract_train_features).
    NOTE: these are NOT used in CV — only in final refit for inference.
    """
    X, labels = [], []
    for _, row in df_spk_test.iterrows():
        spk = clean_speaker(row['Speaker'])
        if spk is None:
            continue
        s = time_to_sec(row['Start'])
        e = time_to_sec(row['End'])
        if (e - s) < 0.2:
            continue
        y_seg = y_full[int(s * sr):int(e * sr)]
        feat  = processor.extract_mfcc(y_seg, sr)
        if feat is not None and len(feat) == 160:
            X.append(feat)
            labels.append(spk)
    return X, labels

def augment_minority(X_real, labels_real, raw_audio, processor, sr=16000):
    """
    Augment minority speakers up to the majority-class count.
    Returns (X_aug, labels_aug) — NOT mixed with real samples.
    Use these only for the final refit, not for CV.
    """
    counts     = Counter(labels_real)
    majority_n = max(counts.values())
    X_aug, labels_aug, aug_counts = [], [], Counter()

    for y_seg, spk in zip(raw_audio, labels_real):
        needed = majority_n - counts[spk]
        if needed <= 0:
            continue
        for y_aug in processor.augment_segment(y_seg, sr):
            if aug_counts[spk] >= needed:
                break
            feat = processor.extract_mfcc(y_aug, sr)
            if feat is not None and len(feat) == 160:
                X_aug.append(feat)
                labels_aug.append(spk)
                aug_counts[spk] += 1

    return X_aug, labels_aug, dict(aug_counts)


def print_feature_summary(counts, aug_counts, X_real, X_all, ref_segs_test):
    print(f"   → Original segments: {dict(counts)}")
    print(f"   → Augmented pool   : {aug_counts} (for refit only, not CV)")
    print(f"   → CV segments      : {len(X_real)} real only ({dict(counts)})")
    print(f"   → Refit segments   : {len(X_all)} real+aug")
    print(f"   → Test DER ref segs: {len(ref_segs_test)}")