import numpy as np, joblib, sys
sys.path.insert(0, r'D:\Final project\c')
from preprocess import AudioProcessor
from moviepy.editor import VideoFileClip

# แปลง mp4 → array ด้วย moviepy
clip = VideoFileClip(r'D:\Final project\result\clip_after_60min.mp4')
audio = clip.audio
sr = 16000
y = audio.to_soundarray(fps=sr)
if y.ndim == 2:
    y = y.mean(axis=1)
y = y.astype(np.float32)
clip.close()
print(f"Loaded: {len(y)/sr:.1f}s")

proc = AudioProcessor()
pipe = joblib.load(r'D:\Final project\models\best_speaker.pkl')

print("\nPredictions at ครูเงาะ zone (63-123s):")
for t in range(63, 123, 3):
    chunk = y[int(t*sr):int((t+0.5)*sr)]
    feat  = proc.extract_mfcc(chunk, sr)
    if feat is not None:
        pred   = pipe.predict([feat])[0]
        proba  = pipe.predict_proba([feat])[0]
        labels = pipe.classes_
        print(f"  t={t:3d}s → {pred:8s}  {dict(zip(labels, proba.round(2)))}")
