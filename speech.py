import os
import time
from pydub import AudioSegment
from google.cloud import speech
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted

GOOGLE_MAX_DURATION_MS = 59_000
MAX_RETRIES            = 3
RETRY_BACKOFF          = 5


class SpeechTranscriber:
    def __init__(self, google_json_path, whisper_model_size=None):
        """
        whisper_model_size : "medium" → load Whisper (used by main.py for research)
                             None     → Google STT only (used by app.py for customers)
        """
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_json_path
        self.google_client = speech.SpeechClient()
        self.whisper_model = None

        if whisper_model_size:
            import whisper as _whisper
            print(f"  [STT] Loading Whisper model ({whisper_model_size})...")
            self.whisper_model = _whisper.load_model(whisper_model_size, device='cpu')
            print(f"  [STT] Whisper ({whisper_model_size}) ready.")
        else:
            print("  [STT] Google STT ready (Whisper disabled).")

    def transcribe_both(self, input_path, temp_prefix,
                        start_ms=0, end_ms=None, label=""):
        audio    = AudioSegment.from_file(input_path)
        end_ms   = min(end_ms if end_ms is not None else len(audio), len(audio))
        offset_s = start_ms / 1000.0
        print(f"\n  [{label}] {start_ms/1000:.1f}s – {end_ms/1000:.1f}s "
              f"({(end_ms-start_ms)/1000:.1f}s)")
        slice_path = f"{temp_prefix}_{label}_slice.wav"
        audio[start_ms:end_ms].export(
            slice_path, format="wav", parameters=["-ar", "16000", "-ac", "1"]
        )

        # ── Google STT (always) ───────────────────────────────────────
        print(f"  [{label}] Google STT...")
        g_segs = self._transcribe_chunks_google(slice_path, offset_s,
                                                temp_prefix, label)
        print(f"  [{label}] Google → {len(g_segs)} segments")

        # ── Whisper (only if model was loaded) ────────────────────────
        w_segs = []
        if self.whisper_model is not None:
            print(f"  [{label}] Whisper...")
            w_segs = self._transcribe_whisper(slice_path, offset_s)
            print(f"  [{label}] Whisper → {len(w_segs)} segments")

        try:
            os.remove(slice_path)
        except OSError:
            pass
        return g_segs, w_segs

    def _transcribe_chunks_google(self, wav_path, offset_sec,
                                  temp_prefix, label):
        audio    = AudioSegment.from_wav(wav_path)
        total_ms = len(audio)
        results  = []
        n_chunks = -(-total_ms // GOOGLE_MAX_DURATION_MS)
        failed   = 0
        for i in range(n_chunks):
            c_start    = i * GOOGLE_MAX_DURATION_MS
            c_end      = min(c_start + GOOGLE_MAX_DURATION_MS, total_ms)
            chunk_path = f"{temp_prefix}_{label}_g{i}.wav"
            audio[c_start:c_end].export(chunk_path, format="wav",
                                        parameters=["-ar", "16000", "-ac", "1"])
            c_offset = offset_sec + c_start / 1000.0
            success, wait, status = False, RETRY_BACKOFF, ""
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    segs    = self._google_chunk(chunk_path, c_offset)
                    results.extend(segs)
                    status  = f"{len(segs)} seg(s)"
                    success = True
                    break
                except ResourceExhausted:
                    if attempt < MAX_RETRIES:
                        print(f"    ↳ Quota, retry in {wait}s "
                              f"({attempt}/{MAX_RETRIES})...")
                        time.sleep(wait); wait *= 2
                    else:
                        status = f"⚠️  QUOTA after {MAX_RETRIES} retries"
                except GoogleAPIError as e:
                    status = f"⚠️  API [{type(e).__name__}]: {e}"; break
                except Exception as e:
                    status = f"⚠️  [{type(e).__name__}]: {e}"; break
            if not success:
                failed += 1
            try:
                os.remove(chunk_path)
            except OSError:
                pass
            print(f"    Google chunk {i+1:>2}/{n_chunks} — {status}")
        if failed:
            print(f"  ⚠️  {failed}/{n_chunks} chunk(s) failed.")
        return results

    def _google_chunk(self, wav_path, offset_sec=0.0):
        with open(wav_path, "rb") as f:
            content = f.read()
        response = self.google_client.recognize(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000, language_code="th-TH",
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True,
            ),
            audio=speech.RecognitionAudio(content=content)
        )
        results = []
        for res in response.results:
            alt = res.alternatives[0]
            chunk_text, start_t, word_buf = "", None, []
            for i, w in enumerate(alt.words):
                if start_t is None:
                    start_t = w.start_time.total_seconds()
                chunk_text += (" " if chunk_text else "") + w.word
                word_buf.append({
                    'word':  w.word,
                    'start': round(offset_sec + w.start_time.total_seconds(), 3),
                    'end':   round(offset_sec + w.end_time.total_seconds(), 3),
                })
                if (i + 1) % 12 == 0 or i == len(alt.words) - 1:
                    results.append({
                        'start': round(offset_sec + start_t, 3),
                        'end':   round(offset_sec + w.end_time.total_seconds(), 3),
                        'text':  chunk_text.strip(),
                        'words': list(word_buf),   # ← word-level timestamps
                    })
                    chunk_text, start_t, word_buf = "", None, []
        return results

    def _transcribe_whisper(self, wav_path, offset_sec=0.0):
        """Only called when self.whisper_model is not None."""
        result = self.whisper_model.transcribe(
            wav_path, language='th',
            fp16=False,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        HALLUCINATION = "การสนทนาภาษาไทย"
        segments = []
        for s in result['segments']:
            text = s['text'].strip()
            if (not text or text == HALLUCINATION
                    or HALLUCINATION in text or len(text) <= 2):
                continue
            words = s.get('words', [])
            if words:
                start = round(offset_sec + words[0]['start'], 3)
                end   = round(offset_sec + words[-1]['end'],  3)
            else:
                start = round(offset_sec + s['start'], 3)
                end   = round(offset_sec + s['end'],   3)
            segments.append({'start': start, 'end': end, 'text': text})
        return segments