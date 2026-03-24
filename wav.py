"""
wav.py — Visualize audio waveforms and spectrograms of speaker segments
         BEFORE they are used in the model.

What it shows:
  1. Full audio waveform with speaker segments colored by speaker
  2. Per-speaker: 3 random segment waveforms + spectrogram
  3. MFCC heatmap per speaker (average of all segments)
  4. Segment duration distribution per speaker

Usage:
    python wav.py
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from collections import defaultdict

# ── CONFIG — change these paths ────────────────────────────────────────
CONFIG = {
    'video':         r"D:\Final project\Data\VideoAndWav\video.mp4",
    'excel_speaker': r"D:\Final project\Data\Label\label_sound.xlsx",
    'ffmpeg_dir':    r"D:\Final project\ffmpeg\ffmpeg-8.0.1-essentials_build\bin",
    'output_dir':    r"D:\Final project\result\wav_plots",
    'train_sec':     3600,   # first 60 min = training data
    'sr':            16000,
}

VALID_SPEAKERS = {'ครูเงาะ', 'ท็อป', 'แชท'}
SPEAKER_MAP    = {'ท๊อป': 'ท็อป'}
COLORS         = {'ครูเงาะ': '#2196F3', 'ท็อป': '#F44336', 'แชท': '#4CAF50'}

# Thai font
_THAI_PROP = None
for _fp in [r'C:\Windows\Fonts\tahoma.ttf',
            '/usr/share/fonts/truetype/freefont/FreeSerif.ttf']:
    if os.path.exists(_fp):
        _THAI_PROP = fm.FontProperties(fname=_fp, size=11)
        break


def _thai(text, size=11):
    if _THAI_PROP:
        return fm.FontProperties(fname=_THAI_PROP.get_file(), size=size)
    return None


def time_to_sec(t_str):
    try:
        p = str(t_str).strip().split('.')
        if len(p) == 3:
            return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
        return float(t_str)
    except Exception:
        return 0.0


def clean_speaker(name):
    s = str(name).strip()
    s = SPEAKER_MAP.get(s, s)
    return s if s in VALID_SPEAKERS else None


def load_audio():
    """Load audio using ffmpeg + librosa."""
    ffmpeg_dir = CONFIG['ffmpeg_dir']
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    from pydub import AudioSegment
    AudioSegment.converter = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    AudioSegment.ffprobe   = os.path.join(ffmpeg_dir, "ffprobe.exe")

    from preprocess import AudioProcessor
    processor = AudioProcessor(ffmpeg_dir)
    print("📂 Loading audio (this may take a minute)...")
    y = processor.load_clean_audio(CONFIG['video'])
    print(f"   Loaded {len(y)/CONFIG['sr']:.1f}s")
    return y


def load_segments(y):
    """Parse excel labels → list of {start, end, speaker, audio}."""
    df = pd.read_excel(CONFIG['excel_speaker'])
    sr = CONFIG['sr']
    train_s = CONFIG['train_sec']
    segments = defaultdict(list)

    for _, row in df.iterrows():
        spk = clean_speaker(row['Speaker'])
        if spk is None:
            continue
        s = time_to_sec(row['Start'])
        e = time_to_sec(row['End'])
        if s >= train_s or (e - s) < 0.2:
            continue
        y_seg = y[int(s * sr):int(e * sr)]
        segments[spk].append({'start': s, 'end': e, 'audio': y_seg,
                               'duration': e - s})

    for spk, segs in segments.items():
        print(f"   {spk}: {len(segs)} segments, "
              f"total {sum(s['duration'] for s in segs):.1f}s, "
              f"avg {np.mean([s['duration'] for s in segs]):.2f}s")
    return segments


# ── Plot 1: Full waveform — MATLAB style ──────────────────────────────
def plot_full_waveform(y, segments):
    sr      = CONFIG['sr']
    train_s = CONFIG['train_sec']
    y_plot  = y[:int(train_s * sr)]
    t       = np.linspace(0, train_s, len(y_plot))

    # ── 1a. Plain waveform (like the reference image) ─────────────────
    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(t, y_plot, color='#1565C0', linewidth=0.4, alpha=0.9)
    ax.set_xlim(0, train_s)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Waveform', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, color='grey')
    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'waveform_plain.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   💾 {path}")

    # ── 1b. Waveform colored by speaker ───────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 4), facecolor='white')
    ax.set_facecolor('white')

    # Draw grey base waveform
    ax.plot(t, y_plot, color='#CCCCCC', linewidth=0.3, alpha=0.7, zorder=1)

    # Overlay colored waveform for each labelled segment
    for spk, segs in segments.items():
        color = COLORS.get(spk, '#999999')
        for seg in segs:
            s_idx = int(seg['start'] * sr)
            e_idx = int(seg['end']   * sr)
            t_seg = t[s_idx:e_idx]
            y_seg = y_plot[s_idx:e_idx]
            if len(t_seg) > 0:
                ax.plot(t_seg, y_seg, color=color, linewidth=0.5,
                        alpha=0.85, zorder=2)

    patches = [mpatches.Patch(color=COLORS.get(s, '#999'), label=s)
               for s in segments]
    patches.append(mpatches.Patch(color='#CCCCCC', label='No label'))
    if _THAI_PROP:
        legend = ax.legend(handles=patches, loc='upper right', fontsize=10)
        for text in legend.get_texts():
            text.set_fontproperties(fm.FontProperties(fname=_THAI_PROP.get_file(), size=10))
    else:
        ax.legend(handles=patches, loc='upper right')

    ax.set_xlim(0, train_s)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel('Time (sec)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Waveform — Colored by Speaker', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, color='grey')
    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'waveform_by_speaker.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   💾 {path}")


# ── Plot 2: Per-speaker sample waveforms + spectrograms ───────────────
def plot_speaker_samples(segments):
    sr = CONFIG['sr']
    for spk, segs in segments.items():
        color = COLORS.get(spk, '#999999')
        # Pick 3 random segments of medium length
        sorted_segs = sorted(segs, key=lambda x: abs(x['duration'] - 3.0))
        samples = sorted_segs[:3]

        fig, axes = plt.subplots(2, 3, figsize=(15, 7))
        spk_label = spk

        for col, seg in enumerate(samples):
            y_seg = seg['audio']
            t     = np.linspace(0, len(y_seg) / sr, len(y_seg))

            # Row 0: Waveform — clean MATLAB style
            axes[0, col].set_facecolor('white')
            axes[0, col].plot(t, y_seg, color=color, linewidth=0.8)
            axes[0, col].set_xlabel('Time (sec)', fontsize=10)
            axes[0, col].set_ylabel('Amplitude', fontsize=10)
            dur_str = f"{seg['start']:.1f}s–{seg['end']:.1f}s ({seg['duration']:.2f}s)"
            axes[0, col].set_title(f"Segment {col+1}\n{dur_str}", fontsize=10)
            axes[0, col].set_ylim(-0.8, 0.8)
            axes[0, col].spines['top'].set_visible(False)
            axes[0, col].spines['right'].set_visible(False)
            axes[0, col].grid(True, linestyle='--', alpha=0.3, color='grey')

            # Row 1: Mel spectrogram
            n_fft = min(len(y_seg), 2048)
            if n_fft >= 64:
                S = librosa.feature.melspectrogram(y=y_seg, sr=sr, n_fft=n_fft,
                                                    hop_length=256, n_mels=80)
                S_db = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_db, sr=sr, hop_length=256,
                                               x_axis='time', y_axis='mel',
                                               ax=axes[1, col], cmap='magma')
                axes[1, col].set_title('Mel Spectrogram', fontsize=10)
                axes[1, col].set_xlabel('Time (s)', fontsize=10)
                axes[1, col].set_ylabel('Hz', fontsize=10)
                plt.colorbar(img, ax=axes[1, col], format='%+2.0f dB')
            else:
                axes[1, col].text(0.5, 0.5, 'Too short',
                                  ha='center', va='center', transform=axes[1, col].transAxes)

        title = f"Speaker: {spk_label} — Sample Waveforms & Spectrograms"
        if _THAI_PROP:
            fig.suptitle(title, fontsize=13, fontweight='bold',
                         fontproperties=_thai(spk_label, 13))
        else:
            fig.suptitle(title, fontsize=13, fontweight='bold')

        plt.tight_layout()
        safe = spk.encode('ascii', 'replace').decode().replace('?', '_')
        path = os.path.join(CONFIG['output_dir'], f'speaker_{safe}_samples.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"   💾 {path}")


# ── Plot 3: Average MFCC heatmap per speaker ──────────────────────────
def plot_mfcc_heatmap(segments):
    sr = CONFIG['sr']
    fig, axes = plt.subplots(1, len(segments), figsize=(6 * len(segments), 5))
    if len(segments) == 1:
        axes = [axes]

    for ax, (spk, segs) in zip(axes, segments.items()):
        all_mfcc = []
        for seg in segs[:50]:   # use up to 50 segments for speed
            y_seg = seg['audio']
            if len(y_seg) < 512:
                continue
            n_fft = min(len(y_seg), 2048)
            mfcc  = librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=40, n_fft=n_fft)
            all_mfcc.append(mfcc)
        if not all_mfcc:
            continue
        # Average MFCC across time frames and segments
        max_frames = max(m.shape[1] for m in all_mfcc)
        padded     = [np.pad(m, ((0, 0), (0, max_frames - m.shape[1])),
                             mode='constant') for m in all_mfcc]
        avg_mfcc   = np.mean(padded, axis=0)

        im = ax.imshow(avg_mfcc, aspect='auto', origin='lower', cmap='coolwarm',
                       interpolation='nearest')
        plt.colorbar(im, ax=ax)
        title = f"Avg MFCC — {spk}\n({len(segs)} segments)"
        if _THAI_PROP:
            ax.set_title(title, fontproperties=_thai(spk, 11))
        else:
            ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time Frames', fontsize=10)
        ax.set_ylabel('MFCC Coefficient', fontsize=10)

    plt.suptitle('Average MFCC Heatmap per Speaker', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'mfcc_heatmap_per_speaker.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 {path}")


# ── Plot 4: Segment duration distribution ─────────────────────────────
def plot_duration_distribution(segments):
    fig, axes = plt.subplots(1, len(segments), figsize=(5 * len(segments), 4),
                             sharey=False)
    if len(segments) == 1:
        axes = [axes]

    for ax, (spk, segs) in zip(axes, segments.items()):
        durations = [s['duration'] for s in segs]
        color     = COLORS.get(spk, '#999999')
        ax.hist(durations, bins=30, color=color, alpha=0.85, edgecolor='white')
        ax.axvline(np.mean(durations), color='black', linestyle='--',
                   linewidth=1.5, label=f"Mean: {np.mean(durations):.2f}s")
        ax.axvline(np.median(durations), color='gray', linestyle=':',
                   linewidth=1.5, label=f"Median: {np.median(durations):.2f}s")
        ax.legend(fontsize=9)
        ax.set_xlabel('Duration (seconds)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        title = f"{spk} ({len(segs)} segs)"
        if _THAI_PROP:
            ax.set_title(title, fontproperties=_thai(spk, 11))
        else:
            ax.set_title(title, fontsize=11)

    plt.suptitle('Segment Duration Distribution per Speaker',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'duration_distribution.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 {path}")


# ── Plot 5: Compare waveform before vs after augmentation ─────────────
def plot_augmentation_example(segments):
    from preprocess import AudioProcessor
    processor = AudioProcessor(CONFIG['ffmpeg_dir'])
    sr        = CONFIG['sr']

    # Use the speaker with fewest segments
    spk = min(segments, key=lambda s: len(segments[s]))
    seg = segments[spk][0]
    y_orig = seg['audio']

    augmented = processor.augment_segment(y_orig, sr)
    aug_labels = ['Pitch +2', 'Pitch −2', 'Time ×0.9', 'Time ×1.1', 'Noise']

    n_aug = min(len(augmented), len(aug_labels))
    fig, axes = plt.subplots(1, n_aug + 1, figsize=(4 * (n_aug + 1), 3))

    t_orig = np.linspace(0, len(y_orig) / sr, len(y_orig))
    axes[0].set_facecolor('white')
    axes[0].plot(t_orig, y_orig, color=COLORS.get(spk, '#999'), linewidth=0.7)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time (sec)'); axes[0].set_ylabel('Amplitude')
    axes[0].set_ylim(-0.8, 0.8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(True, linestyle='--', alpha=0.3, color='grey')

    for i, (y_aug, label) in enumerate(zip(augmented[:n_aug], aug_labels)):
        t_aug = np.linspace(0, len(y_aug) / sr, len(y_aug))
        axes[i + 1].set_facecolor('white')
        axes[i + 1].plot(t_aug, y_aug, color='#FF9800', linewidth=0.7)
        axes[i + 1].set_title(label, fontsize=11)
        axes[i + 1].set_xlabel('Time (sec)')
        axes[i + 1].set_ylim(-0.8, 0.8)
        axes[i + 1].spines['top'].set_visible(False)
        axes[i + 1].spines['right'].set_visible(False)
        axes[i + 1].grid(True, linestyle='--', alpha=0.3, color='grey')

    title = f"Augmentation Examples — {spk}"
    if _THAI_PROP:
        fig.suptitle(title, fontsize=13, fontweight='bold',
                     fontproperties=_thai(spk, 13))
    else:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'augmentation_examples.png')
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 {path}")


# ── Plot 6: Labelled vs Unlabelled regions ────────────────────────────
def plot_label_coverage(y, segments):
    """
    Show which parts of the audio have labels and which don't.
    Top row: full waveform with labeled regions highlighted.
    Bottom row: timeline bar showing labelled (colored) vs unlabelled (grey).
    """
    sr      = CONFIG['sr']
    train_s = CONFIG['train_sec']
    total_s = min(len(y) / sr, train_s)
    t       = np.linspace(0, total_s, int(total_s * sr))
    y_plot  = y[:len(t)]

    # Build label map: array of True/False per second
    label_map  = np.zeros(int(total_s))   # 0 = no label
    spk_map    = {}                        # second → speaker

    for spk, segs in segments.items():
        for seg in segs:
            s_idx = int(seg['start'])
            e_idx = min(int(seg['end']) + 1, int(total_s))
            for i in range(s_idx, e_idx):
                label_map[i] = 1
                spk_map[i]   = spk

    labelled_s   = int(label_map.sum())
    unlabelled_s = int(total_s) - labelled_s
    pct          = labelled_s / int(total_s) * 100

    fig = plt.figure(figsize=(22, 8))
    gs  = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.4)

    # ── Row 0: Waveform ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, y_plot, color='#CCCCCC', linewidth=0.2, alpha=0.8, zorder=1)

    # Shade labelled regions on waveform
    for spk, segs in segments.items():
        color = COLORS.get(spk, '#999999')
        for seg in segs:
            ax0.axvspan(seg['start'], seg['end'],
                        alpha=0.45, color=color, linewidth=0, zorder=2)

    patches = [mpatches.Patch(color=COLORS.get(s, '#999'), label=s)
               for s in segments]
    patches.append(mpatches.Patch(color='#CCCCCC', label='No label'))
    if _THAI_PROP:
        legend = ax0.legend(handles=patches, loc='upper right', fontsize=10)
        for text, spk in zip(legend.get_texts(), list(segments.keys()) + ['No label']):
            text.set_fontproperties(fm.FontProperties(fname=_THAI_PROP.get_file(), size=10))
    else:
        ax0.legend(handles=patches, loc='upper right')

    ax0.set_xlim(0, total_s)
    ax0.set_xlabel('Time (seconds)', fontsize=11)
    ax0.set_ylabel('Amplitude', fontsize=11)
    ax0.set_title(
        f'Audio Waveform — Labelled vs Unlabelled Regions\n'
        f'Labelled: {labelled_s}s ({pct:.1f}%)   '
        f'Unlabelled: {unlabelled_s}s ({100-pct:.1f}%)',
        fontsize=12, fontweight='bold'
    )

    # ── Row 1: Timeline bar (labelled/unlabelled) ──────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.set_xlim(0, total_s)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_title('Label Coverage Timeline', fontsize=11)

    # Draw unlabelled background
    ax1.axhspan(0, 1, color='#EEEEEE', alpha=1.0)

    # Draw each labelled segment as colored bar
    for spk, segs in segments.items():
        color = COLORS.get(spk, '#999999')
        for seg in segs:
            ax1.barh(0.5, seg['end'] - seg['start'],
                     left=seg['start'], height=0.7,
                     color=color, alpha=0.85, linewidth=0)

    # ── Row 2: Per-speaker label density (segments per minute) ────────
    ax2 = fig.add_subplot(gs[2])
    bin_size = 60   # 1-minute bins
    bins     = np.arange(0, total_s + bin_size, bin_size)
    bottom   = np.zeros(len(bins) - 1)

    for spk, segs in segments.items():
        color  = COLORS.get(spk, '#999999')
        counts = np.zeros(len(bins) - 1)
        for seg in segs:
            b = min(int(seg['start'] // bin_size), len(counts) - 1)
            counts[b] += 1
        ax2.bar(bins[:-1], counts, width=bin_size * 0.9, bottom=bottom,
                color=color, alpha=0.85, align='edge', linewidth=0)
        bottom += counts

    ax2.set_xlim(0, total_s)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Segments / min', fontsize=11)
    ax2.set_title('Label Density per Minute (stacked by speaker)', fontsize=11)

    plt.tight_layout()
    path = os.path.join(CONFIG['output_dir'], 'label_coverage.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 {path}")

    # Print summary table
    print(f"\n   Label Coverage Summary:")
    print(f"   {'Speaker':<12} {'Segments':>9} {'Total (s)':>10} {'Coverage':>10}")
    print("   " + "─" * 45)
    for spk, segs in segments.items():
        total_spk = sum(s['duration'] for s in segs)
        print(f"   {spk:<12} {len(segs):>9} {total_spk:>9.1f}s {total_spk/total_s*100:>9.1f}%")
    print(f"   {'All labelled':<12} {sum(len(s) for s in segments.values()):>9} "
          f"{labelled_s:>9}s {pct:>9.1f}%")
    print(f"   {'Unlabelled':<12} {'—':>9} {unlabelled_s:>9}s {100-pct:>9.1f}%")


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"📁 Output → {CONFIG['output_dir']}\n")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    y = load_audio()

    print("\n📋 Loading speaker labels...")
    segments = load_segments(y)

    print("\n🎨 Generating plots...")
    print("  [1/6] Label coverage (labelled vs unlabelled)...")
    plot_label_coverage(y, segments)

    print("  [2/6] Full waveform (plain + colored by speaker)...")
    plot_full_waveform(y, segments)

    print("  [3/6] Per-speaker sample waveforms + spectrograms...")
    plot_speaker_samples(segments)

    print("  [4/6] Average MFCC heatmap per speaker...")
    plot_mfcc_heatmap(segments)

    print("  [5/6] Segment duration distribution...")
    plot_duration_distribution(segments)

    print("  [6/6] Augmentation examples...")
    plot_augmentation_example(segments)

    print(f"\n✅ All plots saved to {CONFIG['output_dir']}")
    print("\nFiles created:")
    for f in sorted(os.listdir(CONFIG['output_dir'])):
        if f.endswith('.png'):
            print(f"  📊 {f}")