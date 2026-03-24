"""
save_models.py — Run this ONCE to:
  1. Train the RF speaker classifier on your labelled data
  2. Save the trained model to models/rf_speaker.pkl

After running this, app.py can work without any labels or training.

Usage:
    python save_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter

# ── ffmpeg path ────────────────────────────────────────────────────────
_FFMPEG_DIR = r"D:\Final project\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
from pydub import AudioSegment as _AS
_AS.converter = os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
_AS.ffprobe   = os.path.join(_FFMPEG_DIR, "ffprobe.exe")

from preprocess  import AudioProcessor
from diarization import SpeakerClassifier
from features    import (time_to_sec, extract_train_features,
                         build_ref_segs_test, augment_minority)

CONFIG = {
    'video':         r"D:\Final project\Data\VideoAndWav\video.mp4",
    'excel_speaker': r"D:\Final project\Data\Label\label_sound.xlsx",
    'excel_content': r"D:\Final project\Data\Label\label_speech.xlsx",
    'ffmpeg_dir':    _FFMPEG_DIR,
    'google_json':   r"D:\Final project\APIk\articulate-life-484709-u6-669311c83c6c.json",
}

MODELS_DIR = r"D:\Final project\models"
TRAIN_S    = 3600.0   # first 60 minutes = training data


def train_and_save():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── 1. Load audio ─────────────────────────────────────────────────
    print("📂 Loading audio...")
    processor = AudioProcessor(CONFIG['ffmpeg_dir'])
    y_full    = processor.load_clean_audio(CONFIG['video'])
    print(f"   Total: {len(y_full)/16000:.1f}s")

    # ── 2. Load speaker labels (train slice only) ─────────────────────
    print("📋 Loading labels...")
    df_speaker  = pd.read_excel(CONFIG['excel_speaker'])
    df_spk_train = df_speaker[
        df_speaker['Start'].apply(time_to_sec) < TRAIN_S
    ].copy()
    print(f"   Train speaker rows: {len(df_spk_train)}")

    # ── 3. Extract features ───────────────────────────────────────────
    print("🔍 Extracting features...")
    X_real, labels_real, raw_audio, _ = extract_train_features(
        df_spk_train, y_full, processor
    )
    counts = Counter(labels_real)
    print(f"   Real segments: {dict(counts)}")

    # ── 4. Augment minority classes ───────────────────────────────────
    print("🔁 Augmenting minority classes...")
    X_aug, labels_aug, aug_counts = augment_minority(
        X_real, labels_real, raw_audio, processor
    )
    print(f"   Augmented: {aug_counts}")

    X_all      = X_real + X_aug
    labels_all = labels_real + labels_aug
    print(f"   Total for training: {len(X_all)} segments")

    # ── 5. Train RF on all data (no CV needed — just best model) ──────
    print("🤖 Training RF classifier on all data...")
    classifier = SpeakerClassifier()

    # Run CV first to confirm accuracy, then refit on all data
    eval_results = classifier.train(X_real, labels_real)
    best_name    = max(
        {k: v for k, v in eval_results.items() if "Baseline" not in k},
        key=lambda k: eval_results[k]['cv_acc']
    )
    rf_acc = eval_results[best_name]['cv_acc']
    print(f"\n   ✅ Best model: {best_name}  CV Acc: {rf_acc:.2%}")

    # Refit on all real + augmented data
    classifier.best_model_name     = best_name
    classifier._inference_pipeline = classifier.pipelines[best_name]
    classifier._inference_pipeline.fit(np.array(X_all), np.array(labels_all))

    # ── 6. Save model ─────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, "rf_speaker.pkl")
    joblib.dump({
        'pipeline':   classifier._inference_pipeline,
        'labels':     classifier.all_labels,
        'model_name': best_name,
        'cv_acc':     rf_acc,
        'speakers':   dict(Counter(labels_all)),
    }, model_path)
    print(f"\n   💾 Model saved → {model_path}")

    # ── 7. Summary ────────────────────────────────────────────────────
    print(f"""
{'='*50}
  MODELS SAVED SUCCESSFULLY
{'='*50}
  RF model  : {model_path}
  CV Acc    : {rf_acc:.2%}
  Speakers  : {classifier.all_labels}
  You can now run app.py on any machine that has
  these model files — no training required.
{'='*50}
""")


if __name__ == "__main__":
    train_and_save()