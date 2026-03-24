import os
import sys
import subprocess
import librosa
import numpy as np
from pydub import AudioSegment

# Suppress CMD popup windows from ffmpeg subprocess calls (Windows only)
if sys.platform == 'win32':
    _orig_popen = subprocess.Popen
    def _silent_popen(*args, **kwargs):
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        return _orig_popen(*args, **kwargs)
    subprocess.Popen = _silent_popen


class AudioProcessor:
    def __init__(self, ffmpeg_path, sr=16000):
        self.sr = sr
        os.environ["PATH"] += os.pathsep + ffmpeg_path
        AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
        AudioSegment.ffprobe   = os.path.join(ffmpeg_path, "ffprobe.exe")

    def load_clean_audio(self, wav_path):
        """
        Convert to wav first if the file is a video/compressed format.
        librosa+soundfile can only read wav/flac/ogg — not mp4/mkv/mp3.
        Using pydub+ffmpeg to convert ensures no audioread fallback.
        """
        import tempfile, os
        ext = os.path.splitext(wav_path)[1].lower()
        if ext not in ('.wav', '.flac', '.ogg'):
            # Convert to a temp wav via pydub/ffmpeg
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            AudioSegment.from_file(wav_path).set_frame_rate(self.sr)                .set_channels(1).export(tmp.name, format='wav')
            y, _ = librosa.load(tmp.name, sr=self.sr)
            os.unlink(tmp.name)
        else:
            y, _ = librosa.load(wav_path, sr=self.sr)
        return librosa.effects.preemphasis(y)

    @staticmethod
    def extract_mfcc(y, sr, n_mfcc=40):
        if y is None or len(y) < 512:
            return None
        n_fft    = min(len(y), 2048)
        if n_fft < 2:
            return None
        n_fft    = n_fft if n_fft % 2 == 0 else n_fft - 1
        mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
        n_frames = mfccs.shape[1]
        width    = min(9, n_frames)
        if width % 2 == 0:
            width -= 1
        if width < 3:
            return np.mean(mfccs, axis=1)   # 40-dim fallback
        delta  = librosa.feature.delta(mfccs, width=width)
        delta2 = librosa.feature.delta(mfccs, order=2, width=width)
        return np.concatenate([
            np.mean(mfccs,  axis=1),
            np.mean(delta,  axis=1),
            np.mean(delta2, axis=1),
            np.std(mfccs,   axis=1),
        ])  # 160-dim

    @staticmethod
    def augment_segment(y, sr):
        augmented = []
        for steps in [2, -2]:
            try:
                augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))
            except Exception:
                pass
        for rate in [0.9, 1.1]:
            try:
                augmented.append(librosa.effects.time_stretch(y, rate=rate))
            except Exception:
                pass
        augmented.append(y + np.random.normal(0, 0.005, len(y)))
        return augmented

    @staticmethod
    def separate_sources_nmf(y, sr, n_components=2, n_fft=2048, hop_length=512):
        """
        NMF-based source separation.
        Separates a mixed audio signal into n_components sources.

        Parameters
        ----------
        y           : np.ndarray  mixed audio signal
        sr          : int         sample rate
        n_components: int         number of sources to separate (default=2)

        Returns
        -------
        sources : list of np.ndarray  separated audio signals
                  length = n_components
        """
        from sklearn.decomposition import NMF

        # ── 1. STFT → magnitude spectrogram ──────────────────────────
        S_full  = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        S_mag   = np.abs(S_full)
        S_phase = np.angle(S_full)

        # ── 2. NMF decomposition: S ≈ W × H ──────────────────────────
        # W = basis  (freq bins × components)
        # H = activations (components × time frames)
        model = NMF(n_components=n_components,
                    init='nndsvda',
                    max_iter=300,
                    random_state=42)
        W = model.fit_transform(S_mag)   # (freq, n_components)
        H = model.components_            # (n_components, time)

        # ── 3. Reconstruct each source via Wiener-like mask ───────────
        S_approx = W @ H                 # reconstructed magnitude
        sources  = []
        for i in range(n_components):
            # soft mask for component i
            mask      = (np.outer(W[:, i], H[i]) /
                         (S_approx + 1e-8))
            S_source  = mask * S_mag * np.exp(1j * S_phase)
            y_source  = librosa.istft(S_source, hop_length=hop_length,
                                      length=len(y))
            sources.append(y_source)
        return sources

    def separate_and_classify(self, y, sr, classifier, top1_thresh=0.60):
        """
        Separate audio into 2 sources via NMF, then classify each.
        Used for overlap windows where 2 speakers may be present.

        Returns
        -------
        result : dict with keys:
            'is_overlap' : bool
            'speakers'   : list of speaker names found
            'sources'    : list of separated audio arrays
        """
        # Only run on short chunks (e.g. 0.5s window)
        sources = self.separate_sources_nmf(y, sr, n_components=2)

        speakers = []
        for src in sources:
            feat = self.extract_mfcc(src, sr)
            if feat is not None and len(feat) == 160:
                spk = classifier.predict(feat)
                speakers.append(spk)

        is_overlap = (len(set(speakers)) == 2)
        return {
            'is_overlap': is_overlap,
            'speakers':   list(set(speakers)),
            'sources':    sources,
        }