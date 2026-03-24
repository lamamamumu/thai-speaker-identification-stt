"""
main.py — Orchestrator for speaker diarization + STT pipeline.

Modules
-------
preprocess.py   Audio loading, MFCC extraction, augmentation
diarization.py  RF/SVM/KNN classifiers, CV, DER computation
speech.py       Google STT + Whisper transcription
features.py     Feature extraction, augmentation, ref-segment building
evaluate.py     WER computation, sliding-window DER, result saving
dialogue.py     Dialogue building and CSV export
"""

import os
import sys
import warnings
import logging
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
warnings.filterwarnings("ignore")
logging.getLogger("sklearn").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from collections import Counter

# ── Fix #5: set ffmpeg path BEFORE pydub loads anywhere ───────────────
_FFMPEG_DIR = r"D:\Final project\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
from pydub import AudioSegment as _pydub_AS
_pydub_AS.converter = os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
_pydub_AS.ffprobe   = os.path.join(_FFMPEG_DIR, "ffprobe.exe")
# ──────────────────────────────────────────────────────────────────────

from preprocess  import AudioProcessor
from diarization import SpeakerClassifier
from speech      import SpeechTranscriber
from features    import (time_to_sec, extract_train_features,
                         augment_minority, print_feature_summary)
from evaluate    import (compute_wer, print_wer_table, save_wer_txt,
                         assign_spk_from_hyp)
from dialogue    import build_dialogue, save_dialogue_csv, save_dialogue_txt

TRAIN_DURATION_MS = 60 * 60 * 1000   # 60 minutes


class DiarizationApp:
    def __init__(self, config):
        self.config      = config
        self.processor   = AudioProcessor(config['ffmpeg_dir'])
        self.classifier  = SpeakerClassifier()
        self.transcriber = SpeechTranscriber(config['google_json'], whisper_model_size='medium')

    def _load_audio(self):
        print("📂 Loading full audio...")
        y_full   = self.processor.load_clean_audio(self.config['video'])
        # Cap to first 60 minutes only
        max_samp = TRAIN_DURATION_MS * 16  # ms → samples (16000 Hz / 1000)
        y_full   = y_full[:max_samp]
        total_ms = int(len(y_full) / 16000 * 1000)
        train_ms = total_ms
        train_s  = train_ms / 1000.0
        has_test = False
        print(f"   Audio  : {total_ms/1000:.1f}s ({total_ms/60000:.1f} min) — train/test via CV")
        return y_full, total_ms, train_ms, train_s, has_test

    def _load_labels(self, train_s):
        df_speaker = pd.read_excel(self.config['excel_speaker'])
        df_content = pd.read_excel(self.config['excel_content'])
        df_spk_train = df_speaker[df_speaker['Start'].apply(time_to_sec) < train_s].copy()
        df_spk_test  = df_speaker[df_speaker['Start'].apply(time_to_sec) >= train_s].copy()
        df_cnt_train = df_content[df_content['Start'].apply(time_to_sec) < train_s].copy()
        df_cnt_test  = df_content[df_content['Start'].apply(time_to_sec) >= train_s].copy()
        print(f"\n   Speaker labels — train: {len(df_spk_train)}, test: {len(df_spk_test)}")
        print(f"   Content labels — train: {len(df_cnt_train)}, test: {len(df_cnt_test)}")
        print(f"   Content words  — train: "
              f"{df_cnt_train['Content'].astype(str).str.split().str.len().sum()}, "
              f"test: {df_cnt_test['Content'].astype(str).str.split().str.len().sum()}")
        return df_speaker, df_content, df_spk_train, df_spk_test, df_cnt_train, df_cnt_test

    def _build_features(self, df_spk_train, y_full):
        print("\n🔍 Extracting features from labelled speaker segments...")
        X_real, labels_real, raw_audio, _ = extract_train_features(
            df_spk_train, y_full, self.processor
        )
        X_aug, labels_aug, aug_counts = augment_minority(
            X_real, labels_real, raw_audio, self.processor
        )
        X_all      = X_real + X_aug
        labels_all = labels_real + labels_aug
        counts_real = Counter(labels_real)
        counts_aug  = dict(aug_counts)
        print_feature_summary(counts_real, aug_counts, X_real, X_all, [])
        return X_real, labels_real, X_all, labels_all, counts_real, counts_aug

    def _train_classifier(self, X_real, labels_real, X_all, labels_all):
        print("\n🤖 Running Stratified 5-Fold CV to select best model...")
        eval_results = self.classifier.train(X_real, labels_real)
        best_name = max(
            {k: v for k, v in eval_results.items() if "Baseline" not in k},
            key=lambda k: eval_results[k]['cv_acc']
        )
        self.classifier.best_model_name     = best_name
        self.classifier._inference_pipeline = self.classifier.pipelines[best_name]

        self.classifier._inference_pipeline.fit(np.array(X_all), np.array(labels_all))
        print(f"\n   ✅ Best model: {best_name} "
              f"(CV Acc = {eval_results[best_name]['cv_acc']:.2%} "
              f"on {len(X_real)} real segments) — "
              f"refit on {len(X_all)} real+aug segments for inference.")
        return eval_results, best_name

    def _transcribe(self, train_ms, total_ms, train_s, has_test, result_dir):
        """
        Fix #4: Cache STT results to JSON so re-runs don't call the API again.
        Delete the cache files to force a fresh transcription.
        """
        import json
        cache_g = os.path.join(result_dir, "cache_google.json")
        cache_w = os.path.join(result_dir, "cache_whisper.json")

        if os.path.exists(cache_g) and os.path.exists(cache_w):
            print(f"\n{'='*20} STT — Loading from cache {'='*20}")
            segs_g = json.load(open(cache_g, encoding='utf-8'))
            segs_w = json.load(open(cache_w, encoding='utf-8'))
            g_train = [s for s in segs_g if s['start'] < train_s]
            g_test  = [s for s in segs_g if s['start'] >= train_s]
            w_train = [s for s in segs_w if s['start'] < train_s]
            w_test  = [s for s in segs_w if s['start'] >= train_s]
            print(f"   ✅ Loaded from cache (delete cache files to re-transcribe)")
        else:
            print(f"\n{'='*20} STT — Full Audio Transcription {'='*20}")
            print("\n[1/2] Train slice (0 – 60 min)...")
            g_train, w_train = self.transcriber.transcribe_both(
                self.config['video'], "temp", start_ms=0, end_ms=train_ms, label="train"
            )
            g_test, w_test = [], []
            if has_test:
                print(f"\n[2/2] Test slice ({train_s:.0f}s – end)...")
                g_test, w_test = self.transcriber.transcribe_both(
                    self.config['video'], "temp",
                    start_ms=train_ms, end_ms=total_ms, label="test"
                )
            # Save cache
            json.dump(g_train + g_test, open(cache_g, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            json.dump(w_train + w_test, open(cache_w, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            print(f"\n   💾 STT results cached → {result_dir}")

        g_all = g_train + g_test
        w_all = w_train + w_test
        print(f"\n   Google total : {len(g_all)} segs (train: {len(g_train)}, test: {len(g_test)})")
        print(f"   Whisper total: {len(w_all)} segs (train: {len(w_train)}, test: {len(w_test)})")
        return g_train, w_train, g_test, w_test, g_all, w_all

    def _print_preview(self, g_all, w_all):
        print(f"\n{'='*20} Transcription Preview (first 15 segs) {'='*20}")
        print(f"  {'Time':<16}  {'Google':<40}  Whisper")
        print("  " + "─" * 100)
        def _best_overlap(seg, candidates):
            best_text, best_ov = "—", 0.0
            for c in candidates:
                ov = min(seg['end'], c['end']) - max(seg['start'], c['start'])
                if ov > best_ov:
                    best_ov = ov
                    best_text = c['text']
            return best_text if best_ov > 0 else "—"
        for g in g_all[:15]:
            t_str  = f"{g['start']:.1f}-{g['end']:.1f}s"
            w_text = _best_overlap(g, w_all)
            print(f"  {t_str:<16}  {g['text'][:38]:<40}  {w_text[:38]}")

    def _print_cv_table(self, eval_results, best_name):
        print(f"\n{'='*20} Cross-Validation Results {'='*20}")
        print(f"\n  {'Model':<26} {'CV Acc':>8} {'±Std':>7}  {'Per-fold':>32}  Beat Baseline?")
        print("  " + "─" * 92)
        baseline_acc = eval_results['Baseline (Majority)']['cv_acc']
        for name, res in eval_results.items():
            is_base  = "Baseline" in name
            beat     = "" if is_base else ("✅" if res['cv_acc'] > baseline_acc else "❌")
            marker   = " ← majority-class" if is_base else ""
            selected = " ⭐ selected" if name == best_name else ""
            folds    = "  ".join(f"{s:.0%}" for s in res['cv_scores'])
            print(f"  {name:<26} {res['cv_acc']:>7.2%} {res['cv_std']:>6.2%}  "
                  f"[{folds}]  {beat}{marker}{selected}")
        print(f"\n  ℹ️  Baseline = {baseline_acc:.2%} (always predicts majority class).")
        print(f"  ⭐ {best_name} selected — refit on all data for inference.")

    def _run_wer(self, df_content, df_cnt_train, df_cnt_test,
                 g_train, w_train, g_test, w_test, g_all, w_all,
                 has_test, result_dir):
        print(f"\n{'='*20} CER — Combined Full Audio (Character-level) {'='*20}")
        print("  (Spaces stripped, character-level comparison — correct for Thai)\n")
        print("  [Train slice only]")
        wer_g_train = compute_wer(df_cnt_train, g_train, "Google ")
        wer_w_train = compute_wer(df_cnt_train, w_train, "Whisper")
        wer_g_test = wer_w_test = None
        if has_test and not df_cnt_test.empty:
            print("\n  [Test slice only]")
            wer_g_test = compute_wer(df_cnt_test, g_test, "Google ")
            wer_w_test = compute_wer(df_cnt_test, w_test, "Whisper")
        print("\n  [Combined — full recording]")
        wer_g_all = compute_wer(df_content, g_all, "Google ")
        wer_w_all = compute_wer(df_content, w_all, "Whisper")
        print_wer_table(wer_g_train, wer_w_train, wer_g_test, wer_w_test,
                        wer_g_all, wer_w_all)
        save_wer_txt(os.path.join(result_dir, "cer_test_detail.txt"),
                     wer_g_test, wer_w_test, df_cnt_test, g_test, w_test)
        return wer_g_train, wer_w_train, wer_g_test, wer_w_test, wer_g_all, wer_w_all

    def _sliding_window_diarize(self, y_audio, start_offset_s, end_s, win=2.0, step=0.5):
        """Slide a window over audio and assign speakers — used for dialogue."""
        SR       = 16000
        pipe     = self.classifier._inference_pipeline
        win_samp = int(win  * SR)
        stp_samp = int(step * SR)
        indices  = list(range(0, len(y_audio) - win_samp + 1, stp_samp))

        def _process(idx):
            feat = self.processor.extract_mfcc(y_audio[idx:idx + win_samp], SR)
            if feat is None:
                return None
            spk = str(pipe.predict([feat])[0])
            return {'start': start_offset_s + idx / SR,
                    'end':   min(start_offset_s + (idx + win_samp) / SR, end_s),
                    'speaker': spk}

        results = [None] * len(indices)
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_process, idx): pos for pos, idx in enumerate(indices)}
            for fut in as_completed(futures):
                pos = futures[fut]
                try:
                    results[pos] = fut.result()
                except Exception:
                    pass

        segs = sorted([r for r in results if r], key=lambda x: x['start'])

        # Merge consecutive same-speaker windows
        merged = []
        for s in segs:
            if merged and merged[-1]['speaker'] == s['speaker'] and s['start'] - merged[-1]['end'] <= step:
                merged[-1]['end'] = s['end']
            else:
                merged.append(dict(s))
        return merged

    def _run_dialogue(self, hyp_g_test, hyp_w_test, g_test, w_test, result_dir):
        if hyp_g_test:
            mg = build_dialogue(hyp_g_test, g_test)
            save_dialogue_csv(os.path.join(result_dir, "dialogue_google_test.csv"), mg)
            save_dialogue_txt(os.path.join(result_dir, "dialogue_google_test.txt"), mg)
        if hyp_w_test:
            mw = build_dialogue(hyp_w_test, w_test)
            save_dialogue_csv(os.path.join(result_dir, "dialogue_whisper_test.csv"), mw)
            save_dialogue_txt(os.path.join(result_dir, "dialogue_whisper_test.txt"), mw)

    def _print_classification_report(self, eval_results, best_name):
        print(f"\n{'='*20} Classification Report — {best_name} (held-out CV) {'='*20}")
        report = eval_results[best_name]['report']
        print(pd.DataFrame(report).transpose().to_string())

        # Fix #3: highlight macro F1 as headline (treats all speakers equally)
        macro_f1 = report['macro avg']['f1-score']
        acc      = report['accuracy'] if isinstance(report['accuracy'], float) else report['accuracy']['f1-score']
        print(f"\n  ⭐ Macro F1  : {macro_f1:.2%}  ← headline metric (equal weight per speaker)")
        print(f"     Accuracy  : {acc:.2%}  (biased toward majority class)")

        print(f"\n[Confusion Matrix — {best_name}]  Labels: {self.classifier.all_labels}")
        print(eval_results[best_name]['conf'])

    def _print_summary(self, eval_results, best_name, cer_g_all, cer_w_all):
        report   = eval_results[best_name]['report']
        macro_f1 = report['macro avg']['f1-score']
        acc      = report['accuracy'] if isinstance(report['accuracy'], float) else report['accuracy']['f1-score']
        baseline = eval_results['Baseline (Majority)']['cv_acc']

        print(f"\n{'='*60}")
        print(f"  FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Speaker Classification  ({best_name}, 5-fold CV on real segments)")
        print(f"    Accuracy   : {acc:.2%}   (baseline: {baseline:.2%})")
        print(f"    Macro F1   : {macro_f1:.2%}   ← recommended headline")
        for spk in self.classifier.all_labels:
            r = report.get(spk, {})
            print(f"    {spk:<10} P={r.get('precision',0):.2%}  "
                  f"R={r.get('recall',0):.2%}  F1={r.get('f1-score',0):.2%}")
        print()
        print(f"  Character Error Rate    (spaces stripped, Thai CER)")
        if cer_g_all is not None:
            print(f"    Google CER : {cer_g_all:.2%}")
        if cer_w_all is not None:
            print(f"    Whisper CER: {cer_w_all:.2%}")
        print(f"{'='*60}")


    # ─────────────────────────────────────────────────────────────────────
    # Save all research results
    # ─────────────────────────────────────────────────────────────────────
    def _plot_audio_pipeline(self, y_full, df_spk_train, result_dir):
        """Save 6 separate pipeline images (one per feature step) per speaker."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import librosa

        SR = 16000
        HOP, N_MELS, N_MFCC = 512, 64, 40
        seg_len_s = 5.0

        SPEAKER_COLORS = {
            'ครูเงาะ': '#2196F3',
            'ท็อป':   '#4CAF50',
            'แชท':    '#FF9800',
        }

        _thai_prop = None
        for _fp2 in [r'C:\Windows\Fonts\tahoma.ttf',
                     '/usr/share/fonts/truetype/freefont/FreeSerif.ttf']:
            if os.path.exists(_fp2):
                _thai_prop = fm.FontProperties(fname=_fp2, size=11)
                break

        from features import time_to_sec, clean_speaker
        df = df_spk_train.copy()
        df['_spk']   = df['Speaker'].apply(clean_speaker)
        df['_start'] = df['Start'].apply(time_to_sec)
        df['_end']   = df['End'].apply(time_to_sec)
        df = df.dropna(subset=['_spk'])
        speakers = sorted(df['_spk'].unique())
        n_spk = len(speakers)

        # pre-compute per-speaker segment data
        segs = {}
        for spk in speakers:
            rows = df[df['_spk'] == spk]
            best, best_len = None, 0
            for _, r in rows.iterrows():
                dur = r['_end'] - r['_start']
                if dur > best_len:
                    best_len, best = dur, r
            if best is None:
                continue
            s     = int(best['_start'] * SR)
            e     = int(min(best['_start'] + seg_len_s, best['_end']) * SR)
            y_seg = y_full[s:e]
            S_mel = librosa.feature.melspectrogram(y=y_seg, sr=SR,
                                                   n_mels=N_MELS, fmax=8000,
                                                   hop_length=HOP)
            mfcc   = librosa.feature.mfcc(y=y_seg, sr=SR, n_mfcc=N_MFCC, hop_length=HOP)
            segs[spk] = {
                'y_seg':  y_seg,
                'y_pre':  librosa.effects.preemphasis(y_seg, coef=0.97),
                'S_mel':  S_mel,
                'S_db':   librosa.power_to_db(S_mel, ref=np.max),
                'mfcc':   mfcc,
                'feat':   np.vstack([mfcc,
                                     librosa.feature.delta(mfcc, order=1),
                                     librosa.feature.delta(mfcc, order=2)]),
                't_wav':  np.linspace(best['_start'],
                                      best['_start'] + len(y_seg) / SR, len(y_seg)),
                't_spec': librosa.times_like(S_mel, sr=SR, hop_length=HOP) + best['_start'],
                'freqs':  librosa.mel_frequencies(n_mels=N_MELS, fmax=8000),
                'colour': SPEAKER_COLORS.get(spk, '#888888'),
            }

        def _make_fig():
            return plt.subplots(1, n_spk, figsize=(6 * n_spk, 4))

        def _title(ax, spk):
            if _thai_prop:
                ax.set_title(spk, fontproperties=_thai_prop, fontsize=11)
            else:
                ax.set_title(spk, fontsize=11)

        steps = [
            ('pipeline_01_waveform.png',         '① Raw Waveform'),
            ('pipeline_02_preemphasis.png',       '② Pre-emphasis (coef=0.97)'),
            ('pipeline_03_mel_spectrogram.png',   '③ Mel-Spectrogram (power)'),
            ('pipeline_04_log_mel_spectrogram.png','④ Log Mel-Spectrogram (dB)'),
            ('pipeline_05_mfcc.png',              '⑤ MFCC (40 coeff)'),
            ('pipeline_06_mfcc_delta.png',        '⑥ MFCC + Δ + ΔΔ (120)'),
        ]

        for step_idx, (fname, step_title) in enumerate(steps):
            fig, axes = _make_fig()
            if n_spk == 1:
                axes = [axes]
            fig.suptitle(step_title, fontsize=13)

            for col, spk in enumerate(speakers):
                if spk not in segs:
                    continue
                d  = segs[spk]
                ax = axes[col]
                _title(ax, spk)

                if step_idx == 0:   # waveform
                    ax.plot(d['t_wav'], d['y_seg'], color=d['colour'], linewidth=0.4)
                    ax.set_xlim(d['t_wav'][0], d['t_wav'][-1])
                    ax.set_ylabel('Amplitude', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.3)

                elif step_idx == 1:  # pre-emphasis
                    ax.plot(d['t_wav'], d['y_pre'], color='#9C27B0', linewidth=0.4)
                    ax.set_xlim(d['t_wav'][0], d['t_wav'][-1])
                    ax.set_ylabel('Amplitude', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.3)

                elif step_idx == 2:  # mel-spectrogram power
                    im = ax.pcolormesh(d['t_spec'], d['freqs'] / 1000, d['S_mel'],
                                       shading='auto', cmap='inferno')
                    ax.set_yscale('symlog', linthresh=0.3)
                    ax.set_ylabel('Freq (kHz)', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    fig.colorbar(im, ax=ax, pad=0.02, label='Power')

                elif step_idx == 3:  # log mel-spectrogram dB
                    im = ax.pcolormesh(d['t_spec'], d['freqs'] / 1000, d['S_db'],
                                       shading='auto', cmap='magma', vmin=-80, vmax=0)
                    ax.set_yscale('symlog', linthresh=0.3)
                    ax.set_ylabel('Freq (kHz)', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    fig.colorbar(im, ax=ax, pad=0.02, format='%+.0f dB')

                elif step_idx == 4:  # MFCC
                    im = ax.pcolormesh(d['t_spec'], np.arange(1, N_MFCC + 1), d['mfcc'],
                                       shading='auto', cmap='coolwarm')
                    ax.set_ylabel('Coeff #', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    fig.colorbar(im, ax=ax, pad=0.02)

                elif step_idx == 5:  # MFCC + Δ + ΔΔ
                    im = ax.pcolormesh(d['t_spec'], np.arange(1, 121), d['feat'],
                                       shading='auto', cmap='coolwarm')
                    for div in [40.5, 80.5]:
                        ax.axhline(div, color='white', linewidth=0.8, linestyle='--')
                    ax.set_ylabel('Feature #', fontsize=9)
                    ax.set_xlabel('Time (s)', fontsize=9)
                    ax.text(d['t_spec'][-1], 20,  'MFCC', ha='right', va='center',
                            fontsize=7, color='white', fontweight='bold')
                    ax.text(d['t_spec'][-1], 60,  'Δ',    ha='right', va='center',
                            fontsize=7, color='white', fontweight='bold')
                    ax.text(d['t_spec'][-1], 100, 'ΔΔ',   ha='right', va='center',
                            fontsize=7, color='white', fontweight='bold')
                    fig.colorbar(im, ax=ax, pad=0.02)

            plt.tight_layout()
            out_path = os.path.join(result_dir, fname)
            fig.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   💾 Pipeline → {fname}")

    def _save_all_results(self, result_dir, eval_results, best_name,
                          cer_g_train, cer_w_train,
                          cer_g_test,  cer_w_test,
                          cer_g_all,   cer_w_all,
                          counts_real, counts_aug,
                          X_real, X_all):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        def fp(v): return round(v, 4) if v is not None else None
        labels = self.classifier.all_labels

        _thai_prop = None
        for _fp2 in [r'C:\Windows\Fonts\tahoma.ttf',
                     '/usr/share/fonts/truetype/freefont/FreeSerif.ttf']:
            if os.path.exists(_fp2):
                _thai_prop = fm.FontProperties(fname=_fp2, size=13)
                break

        # ── 1. Confusion Matrix per model ────────────────────────────
        for name, res in eval_results.items():
            conf = res['conf']
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(conf, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            if _thai_prop:
                ax.set_xticklabels(labels, fontproperties=_thai_prop, rotation=30, ha='right')
                ax.set_yticklabels(labels, fontproperties=_thai_prop)
            else:
                ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=13)
                ax.set_yticklabels(labels, fontsize=13)
            thresh = conf.max() / 2
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, str(conf[i, j]), ha='center', va='center',
                            color='white' if conf[i, j] > thresh else 'black',
                            fontsize=14, fontweight='bold')
            acc_v    = res['cv_acc']
            macro_f1 = res.get('macro_f1', 0)
            title    = f"Confusion Matrix — {name}\nCV Acc: {acc_v:.2%}  |  Macro F1: {macro_f1:.2%}"
            if _thai_prop:
                ax.set_title(title, fontproperties=fm.FontProperties(fname=_thai_prop.get_file(), size=12))
            else:
                ax.set_title(title, fontsize=12)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            plt.tight_layout()
            safe = name.replace(' ', '_').replace('(', '').replace(')', '')
            img_path = os.path.join(result_dir, f"confusion_matrix_{safe}.png")
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   💾 Conf Mat → {img_path}")

        # ── 2. Classification report CSV per model ────────────────────
        for name, res in eval_results.items():
            rows = []
            for cls in labels + ['macro avg', 'weighted avg']:
                r = res['report'].get(cls, {})
                rows.append({'Class': cls,
                    'Precision': fp(r.get('precision')), 'Recall': fp(r.get('recall')),
                    'F1': fp(r.get('f1-score')), 'Support': int(r.get('support', 0))})
            safe = name.replace(' ', '_').replace('(', '').replace(')', '')
            pd.DataFrame(rows).to_csv(
                os.path.join(result_dir, f"classification_report_{safe}.csv"),
                index=False, encoding='utf-8-sig')
            print(f"   💾 Cls Rpt  → classification_report_{safe}.csv")

        # ── 3. CER CSV ────────────────────────────────────────────────
        pd.DataFrame([
            {'Engine': 'Google',  'Train': fp(cer_g_train), 'Test': fp(cer_g_test), 'Combined': fp(cer_g_all)},
            {'Engine': 'Whisper', 'Train': fp(cer_w_train), 'Test': fp(cer_w_test), 'Combined': fp(cer_w_all)},
        ]).to_csv(os.path.join(result_dir, "cer_results.csv"), index=False, encoding='utf-8-sig')
        print(f"   💾 CER      → cer_results.csv")

        # ── 4. Data count CSV ─────────────────────────────────────────
        count_rows = []
        for spk in labels:
            count_rows.append({'Speaker': spk,
                'Before_Augment': counts_real.get(spk, 0),
                'Augmented':      counts_aug.get(spk, 0),
                'After_Augment':  counts_real.get(spk, 0) + counts_aug.get(spk, 0)})
        count_rows.append({'Speaker': 'TOTAL',
            'Before_Augment': len(X_real),
            'Augmented':      sum(counts_aug.values()),
            'After_Augment':  len(X_all)})
        pd.DataFrame(count_rows).to_csv(
            os.path.join(result_dir, "data_count.csv"), index=False, encoding='utf-8-sig')
        print(f"   💾 Data Cnt → data_count.csv")

        # ── 6. All results Excel ──────────────────────────────────────
        xlsx_path = os.path.join(result_dir, "research_results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            cv_rows = []
            for name, res in eval_results.items():
                row = {'Model': name, 'CV_Acc': res['cv_acc'], 'CV_Std': res['cv_std'],
                       'Macro_F1': res.get('macro_f1', 0)}
                for i, s in enumerate(res['cv_scores']):
                    row[f'Fold{i+1}'] = s
                cv_rows.append(row)
            pd.DataFrame(cv_rows).to_excel(writer, sheet_name='CV_Results', index=False)
            pd.DataFrame([
                {'Engine': 'Google',  'Train': cer_g_train, 'Test': cer_g_test, 'Combined': cer_g_all},
                {'Engine': 'Whisper', 'Train': cer_w_train, 'Test': cer_w_test, 'Combined': cer_w_all},
            ]).to_excel(writer, sheet_name='CER', index=False)
            cls_rows = []
            for cls in labels + ['macro avg', 'weighted avg']:
                r = eval_results[best_name]['report'].get(cls, {})
                cls_rows.append({'Model': best_name, 'Class': cls,
                    'Precision': r.get('precision'), 'Recall': r.get('recall'),
                    'F1': r.get('f1-score'), 'Support': r.get('support', 0)})
            pd.DataFrame(cls_rows).to_excel(writer, sheet_name='Classification', index=False)
            pd.DataFrame(count_rows).to_excel(writer, sheet_name='Data_Count', index=False)
        print(f"   💾 Excel     → research_results.xlsx")
        print(f"\n   ✅ All results saved to {result_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────
    def run(self):
        try:
            result_dir = self.config.get(
                'result_dir',
                os.path.join(os.path.dirname(self.config['video']), 'result')
            )
            os.makedirs(result_dir, exist_ok=True)
            print(f"📁 Results will be saved to: {result_dir}")

            y_full, total_ms, train_ms, train_s, has_test = self._load_audio()

            (df_speaker, df_content,
             df_spk_train, _,
             df_cnt_train, df_cnt_test) = self._load_labels(train_s)

            (X_real, labels_real,
             X_all, labels_all,
             counts_real, counts_aug) = self._build_features(df_spk_train, y_full)

            self._plot_audio_pipeline(y_full, df_spk_train, result_dir)

            eval_results, best_name = self._train_classifier(
                X_real, labels_real, X_all, labels_all
            )

            g_train, w_train, g_test, w_test, g_all, w_all = self._transcribe(
                train_ms, total_ms, train_s, has_test, result_dir
            )

            self._print_preview(g_all, w_all)
            self._print_cv_table(eval_results, best_name)
            (cer_g_train, cer_w_train,
             cer_g_test,  cer_w_test,
             cer_g_all,   cer_w_all) = self._run_wer(
                df_content, df_cnt_train, df_cnt_test,
                g_train, w_train, g_test, w_test, g_all, w_all,
                has_test, result_dir
            )

            self._print_classification_report(eval_results, best_name)
            self._print_summary(eval_results, best_name, cer_g_all, cer_w_all)

            print(f"\n{'='*20} Speaker Diarization (full audio) {'='*20}")
            end_s    = len(y_full) / 16000
            # Filter STT to match capped audio (60 min) — avoids Unknown from stale cache
            g_dlg = [s for s in g_all if s['start'] < end_s]
            w_dlg = [s for s in w_all if s['start'] < end_s]
            hyp_segs = self._sliding_window_diarize(y_full, 0.0, end_s)
            hyp_g    = assign_spk_from_hyp(g_dlg, hyp_segs)
            hyp_w    = assign_spk_from_hyp(w_dlg, hyp_segs)
            self._run_dialogue(hyp_g, hyp_w, g_dlg, w_dlg, result_dir)

            print(f"\n{'='*20} Saving Results {'='*20}")
            self._save_all_results(
                result_dir, eval_results, best_name,
                cer_g_train, cer_w_train,
                cer_g_test,  cer_w_test,
                cer_g_all,   cer_w_all,
                counts_real, counts_aug,
                X_real, X_all,
            )

            # ── Save best model .pkl ──────────────────────────────────
            model_dir = self.config.get(
                'model_dir',
                os.path.normpath(os.path.join(os.path.dirname(self.config['video']), '..', 'models'))
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_speaker.pkl')
            joblib.dump({
                'pipeline':   self.classifier._inference_pipeline,
                'labels':     self.classifier.all_labels,
                'model_name': best_name,
                'cv_acc':     eval_results[best_name]['cv_acc'],
                'macro_f1':   eval_results[best_name].get('macro_f1', 0),
                'speakers':   dict(Counter(labels_all)),
            }, model_path)
            print(f"\n   💾 Best model ({best_name}) saved \u2192 {model_path}")

        except KeyboardInterrupt:
            print("\n⛔ Stopped by user.")
        except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    CONFIG = {
        'video':         r"",
        'excel_speaker': r"",
        'excel_content': r"",
        'ffmpeg_dir':    r"",
        'google_json':   r"",
        'result_dir':    r"",
        'model_dir':     r"",
    }
    app = DiarizationApp(CONFIG)
    app.run()