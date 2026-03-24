import os
import pandas as pd
import numpy as np
import librosa
import whisper
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from google.cloud import speech

# ==========================================
# 1. Class: Audio Processor
# ==========================================
class AudioProcessor:
    def __init__(self, ffmpeg_path, sr=16000):
        self.sr = sr
        os.environ["PATH"] += os.pathsep + ffmpeg_path
        AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
        AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

    def trim_and_export(self, input_path, output_path, duration_ms):
        audio = AudioSegment.from_file(input_path)
        trimmed = audio[:duration_ms]
        trimmed.export(output_path, format="wav", parameters=["-ar", str(self.sr), "-ac", "1"])
        return self.load_clean_audio(output_path)

    def load_clean_audio(self, wav_path):
        y, _ = librosa.load(wav_path, sr=self.sr)
        return librosa.effects.preemphasis(y)

    @staticmethod
    def extract_mfcc(y, sr, n_mfcc=40):
        if len(y) < 512: return None # ป้องกัน Segment สั้นเกินไป
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)

# ==========================================
# 2. Class: Speaker Classifier (ML Models)
# ==========================================
class SpeakerClassifier:
    def __init__(self):
        self.models = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True),
            'KNN': KNeighborsClassifier(n_neighbors=3)
        }

    def train(self, X, y):
        print(f"--- Training models with {len(X)} samples ---")
        for name, model in self.models.items():
            model.fit(X, y)
            print(f"Model {name} trained.")

    def predict_all(self, feature):
        return {name: model.predict([feature])[0] for name, model in self.models.items()}

# ==========================================
# 3. Class: Speech Transcriber (Dual Engine)
# ==========================================
class SpeechTranscriber:
    def __init__(self, google_json_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_json_path
        self.google_client = speech.SpeechClient()
        self.whisper_model = whisper.load_model("base")

    def transcribe_whisper(self, wav_path):
        print("--- Transcribing with Whisper ---")
        return self.whisper_model.transcribe(wav_path, language='th')['segments']

    def transcribe_google(self, wav_path):
            print("--- Transcribing with Google ---")
            with open(wav_path, "rb") as f:
                content = f.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000, 
                language_code="th-TH", 
                enable_word_time_offsets=True, # สำคัญมากสำหรับการตัดแบ่งเอง
                enable_automatic_punctuation=True 
            )
            
            response = self.google_client.recognize(config=config, audio=audio)
            results = []
            
            for res in response.results:
                alt = res.alternatives[0]
                
                # --- Logic: ตัดแบ่งประโยคยาวให้สั้นลง (เช่น ทุกๆ 10 คำ หรือเจอจุดพัก) ---
                chunk_text = ""
                start_t = None
                
                for i, word_info in enumerate(alt.words):
                    if start_t is None:
                        start_t = word_info.start_time.total_seconds()
                    
                    chunk_text += word_info.word
                    
                    # เงื่อนไขการตัดแบ่ง: ทุกๆ 12 คำ หรือเป็นคำสุดท้าย
                    if (i + 1) % 12 == 0 or (i == len(alt.words) - 1):
                        end_t = word_info.end_time.total_seconds()
                        results.append({
                            'start': start_t,
                            'end': end_t,
                            'text': chunk_text.strip()
                        })
                        # รีเซ็ตเพื่อเริ่มประโยคย่อยใหม่
                        chunk_text = ""
                        start_t = None
            return results

# ==========================================
# 4. Main Controller
# ==========================================
class DiarizationApp:
    def __init__(self, config):
        self.config = config
        self.processor = AudioProcessor(config['ffmpeg_dir'])
        self.classifier = SpeakerClassifier()
        self.transcriber = SpeechTranscriber(config['google_json'])
        self.temp_wav = "temp_process.wav"

    def time_to_sec(self, t_str):
        p = str(t_str).replace('.', ':').split(':')
        return float(p[0])*3600 + float(p[1])*60 + float(p[2]) if len(p)==3 else 0.0

    def run(self):
        try:
            # Step 1: Prep & Train
            y_clean = self.processor.trim_and_export(self.config['video'], self.temp_wav, 59000)
            
            df = pd.read_excel(self.config['excel'])
            X, labels = [], []
            for _, row in df.iterrows():
                times = str(row['เวลา']).split('–')
                if len(times) < 2: continue
                s, e = self.time_to_sec(times[0]), self.time_to_sec(times[1])
                feat = self.processor.extract_mfcc(y_clean[int(s*16000):int(e*16000)], 16000)
                if feat is not None:
                    X.append(feat); labels.append(row['คนพูด'])
            
            self.classifier.train(X, labels)

            # Step 2: Transcribe & Compare
            w_segs = self.transcriber.transcribe_whisper(self.temp_wav)
            g_segs = self.transcriber.transcribe_google(self.temp_wav)

            self.display_results("GOOGLE STT", g_segs, y_clean)
            self.display_results("WHISPER STT", w_segs, y_clean)

        except Exception as e: print(f"❌ Error: {e}")
        finally: 
            if os.path.exists(self.temp_wav): os.remove(self.temp_wav)

    def display_results(self, title, segments, y):
        print(f"\n{'='*20} {title} {'='*20}")
        print(f"{'Time':<15} | {'RF':<10} | {'SVM':<10} | {'KNN':<10} | {'Text'}")
        for seg in segments:
            feat = self.processor.extract_mfcc(y[int(seg['start']*16000):int(seg['end']*16000)], 16000)
            if feat is not None:
                preds = self.classifier.predict_all(feat)
                print(f"{seg['start']:>5.2f}-{seg['end']:>5.2f}s | {preds['RF']:<10} | {preds['SVM']:<10} | {preds['KNN']:<10} | {seg['text']}")

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    CONFIG = {
        'video': r"D:\Final project\Data\VideoAndWav\video.mp4",
        'excel': r"D:\Final project\Data\Label\label_sound.xlsx",
        'ffmpeg_dir': r"C:\Users\usEr\AppData\Local\Overwolf\Extensions\ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb\270.0.25\obs\bin\64bit",
        'google_json': r"D:\Final project\APIk\articulate-life-484709-u6-669311c83c6c.json"
    }
    app = DiarizationApp(CONFIG)
    app.run()

