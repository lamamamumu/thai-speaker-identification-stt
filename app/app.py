"""
app.py — Customer-facing GUI for Speaker Diarization (Google STT only).
No training required. Loads pre-trained model from best_speaker.pkl (saved by main.py).
"""

import os, sys, subprocess

# ── Suppress CMD popup windows from subprocess calls (Windows only) ───
if sys.platform == 'win32':
    _orig_popen = subprocess.Popen
    def _silent_popen(*args, **kwargs):
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
        return _orig_popen(*args, **kwargs)
    subprocess.Popen = _silent_popen

# ── Must be set BEFORE any other import ───────────────────────────────
os.environ['MPLBACKEND'] = 'Agg'
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import json, joblib
import numpy as np

# ── Base path — defined FIRST before any path is used ─────────────────
def _base():
    """Return folder containing Designer/models/APIk/ffmpeg.
    - As .py script: Designer is one level up from c/ folder
    - As .exe: PyInstaller 6+ puts files in _internal/ subfolder
    """
    if getattr(sys, 'frozen', False):
        exe_dir      = os.path.dirname(sys.executable)
        internal_dir = os.path.join(exe_dir, "_internal")
        if os.path.exists(os.path.join(internal_dir, "Designer")):
            return internal_dir
        return exe_dir
    # Running as .py — Designer/models/APIk are one level up from this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if os.path.exists(os.path.join(parent_dir, "Designer")):
        return parent_dir
    return script_dir   # fallback: same folder as script

BASE = _base()

# ── ffmpeg ─────────────────────────────────────────────────────────────
_FFMPEG_DIR = os.path.join(BASE, "ffmpeg", "ffmpeg-8.0.1-essentials_build", "bin")
os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
from pydub import AudioSegment as _AS
_AS.converter = os.path.join(_FFMPEG_DIR, "ffmpeg.exe")
_AS.ffprobe   = os.path.join(_FFMPEG_DIR, "ffprobe.exe")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QTableWidgetItem, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui  import QColor, QFont
from PyQt6        import uic

# ── Config — paths relative to exe/script ─────────────────────────────
MODEL_PATH  = os.path.join(BASE, "models",   "best_speaker.pkl")
RESULT_DIR  = os.path.join(BASE, "result")
UI_PATH     = os.path.join(BASE, "Designer", "main_designer.ui")
GOOGLE_JSON = os.path.join(BASE, "APIk",     "articulate-life-484709-u6-669311c83c6c.json")
SPEAKER_COLORS = {
    'ครูเงาะ': '#d0e8ff',
    'ท็อป':   '#d4f0d4',
    'แชท':    '#fff0d0',
}


# ──────────────────────────────────────────────────────────────────────
# Inference Engine
# ──────────────────────────────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, log_fn=None):
        self.log = log_fn or print
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun main.py or save_models.py first.")
        data            = joblib.load(MODEL_PATH)
        self.pipeline   = data['pipeline']
        self.labels     = data['labels']
        self.model_name = data['model_name']
        self.cv_acc     = data['cv_acc']
        macro_f1        = data.get('macro_f1', None)
        f1_str          = f"  Macro F1: {macro_f1:.2%}" if macro_f1 else ""
        self.log(f" Model: {self.model_name}  CV Acc: {self.cv_acc:.2%}{f1_str}")
        self.log(f"   Speakers: {self.labels}")

    def run(self, video_path, progress_fn=None, stop_fn=None):
        os.environ['MPLBACKEND'] = 'Agg'   # ensure Agg inside thread too
        from preprocess import AudioProcessor
        from speech     import SpeechTranscriber
        from evaluate   import assign_spk_from_hyp
        from dialogue   import build_dialogue


        def step(n, msg):
            self.log(f"{msg}  ({n}%)")
            if progress_fn: progress_fn(n)

        def check_stop():
            if stop_fn and stop_fn():
                raise InterruptedError("Stopped by user.")

        os.makedirs(RESULT_DIR, exist_ok=True)

        # 1. Load audio
        step(5, " Loading audio...")
        processor = AudioProcessor(_FFMPEG_DIR)
        y_full    = processor.load_clean_audio(video_path)
        total_s   = len(y_full) / 16000
        total_ms  = int(total_s * 1000)
        self.log(f"   Duration: {total_s:.1f}s  ({total_s/60:.1f} min)")
        check_stop()

        # 2. Google STT with smart cache
        step(10, " Transcribing with Google STT...")
        cache_g    = os.path.join(RESULT_DIR, "cache_google.json")
        cache_meta = os.path.join(RESULT_DIR, "cache_meta.json")

        def _cache_valid():
            if not os.path.exists(cache_g) or not os.path.exists(cache_meta):
                return False
            try:
                meta = json.load(open(cache_meta, encoding='utf-8'))
                return (meta.get('video_path') == video_path and
                        abs(meta.get('duration_s', 0) - total_s) < 1.0)
            except Exception:
                return False

        if not _cache_valid():
            for f in [cache_g, cache_meta]:
                if os.path.exists(f): os.remove(f)
            self.log("     Cache miss — calling Google STT...")
            transcriber = SpeechTranscriber(GOOGLE_JSON)
            step(15, "   Transcribing full file...")
            segs_g, _ = transcriber.transcribe_both(
                video_path, os.path.join(RESULT_DIR, "temp"), start_ms=0, end_ms=total_ms, label="full"
            )
            json.dump(segs_g, open(cache_g, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            json.dump({'video_path': video_path, 'duration_s': total_s},
                      open(cache_meta, 'w', encoding='utf-8'))
            self.log(f"    Cached {len(segs_g)} segments")
        else:
            segs_g = json.load(open(cache_g, encoding='utf-8'))
            self.log(f"    Loaded {len(segs_g)} segments from cache")

        self.log(f"   Total segments: {len(segs_g)}")
        check_stop()

        # 3. Sliding-window diarization — full file, no split
        step(40, " Running speaker diarization...")
        pipeline = self.pipeline
        SR       = 16000
        win_s, step_s = 2.0, 0.5
        win_samp = int(win_s  * SR)
        stp_samp = int(step_s * SR)
        indices  = list(range(0, len(y_full) - win_samp + 1, stp_samp))

        def _process(idx):
            feat = processor.extract_mfcc(y_full[idx:idx + win_samp], SR)
            if feat is None:
                return None
            spk = str(pipeline.predict([feat])[0])
            return {'start': idx / SR,
                    'end':   min((idx + win_samp) / SR, total_s),
                    'speaker': spk}

        results = []
        for idx in indices:
            r = _process(idx)
            if r is not None:
                results.append(r)

        segs = sorted(results, key=lambda x: x['start'])
        hyp  = []
        for s in segs:
            if hyp and hyp[-1]['speaker'] == s['speaker'] and s['start'] - hyp[-1]['end'] <= step_s:
                hyp[-1]['end'] = s['end']
            else:
                hyp.append(dict(s))

        self.log(f"   Diarization: {len(hyp)} segments")
        check_stop()

        # 4. Assign speakers
        step(80, " Merging speakers with transcript...")
        hyp_g = assign_spk_from_hyp(segs_g, hyp)

        # 5. Build dialogue
        step(90, " Building dialogue...")

        def sec_to_hmmss(s):
            s = int(s)
            return f"{s//3600}.{(s%3600)//60:02d}.{s%60:02d}"

        rows = []
        for d in build_dialogue(hyp_g, segs_g):
            rows.append({
                'Start':   sec_to_hmmss(d['start']),
                'End':     sec_to_hmmss(d['end']),
                'Speaker': d['speaker'],
                'Text':    d['text'].replace(' ', ''),
            })

        step(100, f" Done — {len(rows)} turns")
        # Clean up leftover temp wav files
        import glob
        for _f in glob.glob(os.path.join(RESULT_DIR, "temp*.wav")):
            try:
                os.remove(_f)
            except Exception:
                pass

        return rows


# ──────────────────────────────────────────────────────────────────────
# Worker thread
# ──────────────────────────────────────────────────────────────────────
class Worker(QObject):
    log        = pyqtSignal(str)
    progress   = pyqtSignal(int)
    rows_ready = pyqtSignal(list)
    finished   = pyqtSignal()
    error      = pyqtSignal(str)

    def __init__(self, engine, video_path):
        super().__init__()
        self.engine          = engine
        self.video_path      = video_path
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        os.environ['MPLBACKEND'] = 'Agg'

        # Route engine.log through signal (thread-safe) instead of calling
        # MainWindow.log() directly from the worker thread (Qt thread violation)
        _orig_log = self.engine.log
        self.engine.log = self.log.emit

        try:
            rows = self.engine.run(
                self.video_path,
                progress_fn=self.progress.emit,
                stop_fn=lambda: self._stop_requested
            )
            self.engine.log = _orig_log
            self.rows_ready.emit(rows)
            self.finished.emit()
        except InterruptedError as e:
            self.engine.log = _orig_log
            self.log.emit(f"⏹ {e}")
            self.finished.emit()
        except Exception:
            self.engine.log = _orig_log
            import traceback
            tb = traceback.format_exc()
            try: self.log.emit(" Error:\n" + tb)
            except Exception: pass
            try: self.error.emit(tb)
            except Exception: pass


# ──────────────────────────────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_PATH, self)
        self.setWindowTitle("Speaker Diarization")

        self._rows   = []
        self._thread = None
        self._worker = None
        self._engine = None

        self._setup_table()
        self._connect_signals()
        self._set_save_enabled(False)
        self.progressBar.setValue(0)

        # Load model 200ms after window appears
        QTimer.singleShot(200, self._load_model_deferred)
        self.log("Starting up...")

    # ── Model ────────────────────────────────────────────────────────
    def _load_model_deferred(self):
        try:
            self._engine = InferenceEngine(log_fn=self.log)
            self.log("─" * 50)
            self.log("Ready — select a video or audio file.")
        except FileNotFoundError as e:
            self.log(f"  {e}")
            self.log("Run save_models.py first.")
        except Exception:
            import traceback
            self.log("  Model error:\n" + traceback.format_exc())

    # ── Table ────────────────────────────────────────────────────────
    def _setup_table(self):
        t = self.tableWidget_output
        t.setColumnCount(4)
        t.setHorizontalHeaderLabels(['Start', 'End', 'Speaker', 'Text'])
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(t.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(t.SelectionBehavior.SelectRows)
        t.setFont(QFont("Segoe UI", 9))
        t.setColumnWidth(0, 80)
        t.setColumnWidth(1, 80)
        t.setColumnWidth(2, 100)

    # ── Signals ──────────────────────────────────────────────────────
    def _connect_signals(self):
        self.pushButton_path.clicked.connect(self._browse)
        self.pushButton_txt.clicked.connect(self._save_txt)
        self.pushButton_csv.clicked.connect(self._save_csv)
        self.pushButton_reset.clicked.connect(self._reset)

    # ── Thread management ────────────────────────────────────────────
    def _cleanup_thread(self):
        # Disconnect worker signals first
        if self._worker is not None:
            try:
                self._worker.log.disconnect()
                self._worker.progress.disconnect()
                self._worker.rows_ready.disconnect()
                self._worker.finished.disconnect()
                self._worker.error.disconnect()
            except Exception:
                pass
        # Stop thread and wait for it to fully finish
        if self._thread is not None:
            try:
                if self._thread.isRunning():
                    self._thread.quit()
                    if not self._thread.wait(5000):   # wait up to 5s
                        self._thread.terminate()       # force kill if stuck
                        self._thread.wait(1000)
            except Exception:
                pass
        # Clear references AFTER thread stops
        self._worker = None
        self._thread = None

    # ── Browse & Run ─────────────────────────────────────────────────
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video / Audio File",
            r"D:\Final project\Data\VideoAndWav",
            "Media Files (*.mp4 *.mp3 *.wav *.m4a *.mkv *.avi *.mov *.flac *.ogg *.webm)"
            ";;All Files (*)"
        )
        if not path:
            return
        self.lineEdit_path.setText(path)
        self._run(path)

    def _run(self, path):
        if self._thread and self._thread.isRunning():
            QMessageBox.warning(self, "Busy",
                "Pipeline is still running.\nClick Reset first.")
            return
        if self._engine is None:
            QMessageBox.critical(self, "No model",
                "Model not loaded.\nRun save_models.py first.")
            return

        self._cleanup_thread()
        self.plainTextEdit_log.clear()
        self.tableWidget_output.setRowCount(0)
        self._rows.clear()
        self._set_save_enabled(False)
        self.progressBar.setValue(0)

        self._thread = QThread()
        self._worker = Worker(self._engine, path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self.log)
        self._worker.progress.connect(self.progressBar.setValue)
        self._worker.rows_ready.connect(self._add_rows)
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        # Keep worker alive until thread fully stops, then schedule cleanup
        self._thread.finished.connect(lambda: None)  # hold reference
        self._thread.start()

    # ── Reset ────────────────────────────────────────────────────────
    def _reset(self):
        if self._worker is not None:
            self._worker.stop()
        sys.stdout = sys.__stdout__
        self._cleanup_thread()
        self.lineEdit_path.clear()
        self.plainTextEdit_log.clear()
        self.tableWidget_output.setRowCount(0)
        self._rows.clear()
        self.progressBar.setValue(0)
        self._set_save_enabled(False)
        self.log(" Reset complete — select a new file.")

    # ── Slots ────────────────────────────────────────────────────────
    def _add_rows(self, rows):
        try:
            t = self.tableWidget_output
            t.setUpdatesEnabled(False)
            for row in rows:
                if row.get('Speaker') == 'Unknown':
                    continue
                self._rows.append(row)
                idx = t.rowCount()
                t.insertRow(idx)
                color = QColor(SPEAKER_COLORS.get(row.get('Speaker', ''), '#ffffff'))
                for col, key in enumerate(['Start', 'End', 'Speaker', 'Text']):
                    item = QTableWidgetItem(str(row.get(key, '')))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter)
                    item.setBackground(color)
                    t.setItem(idx, col, item)
            t.setUpdatesEnabled(True)
        except Exception as e:
            print(f"[_add_rows] {e}")

    def _done(self):
        try:
            self.log(" Complete.")
            self._set_save_enabled(True)
            self.progressBar.setValue(100)
            self.tableWidget_output.scrollToTop()
        except Exception as e:
            print(f"[_done] {e}")

    def _err(self, tb):
        try:
            self.log(" Error:\n" + tb)
            self.progressBar.setValue(0)
            QMessageBox.warning(self, "Error", "See Operation Log for details.")
        except Exception as e:
            print(f"[_err] {e}")

    def log(self, msg):
        try:
            self.plainTextEdit_log.appendPlainText(str(msg))
            sb = self.plainTextEdit_log.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception as e:
            print(f"[log] {e}")

    # ── Save ─────────────────────────────────────────────────────────
    def _save_txt(self):
        if not self._rows:
            QMessageBox.warning(self, "No data", "Run the pipeline first.")
            return
        from datetime import datetime
        stem     = os.path.splitext(os.path.basename(self.lineEdit_path.text()))[0]
        default  = os.path.join(RESULT_DIR, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save Text File", default, "Text Files (*.txt)"
        )
        if not dest:
            return
        col_w  = [10, 10, 12]
        header = f"{'Start':<{col_w[0]}} {'End':<{col_w[1]}} {'Speaker':<{col_w[2]}} Text"
        lines  = [header, "-" * 80]
        for row in self._rows:
            lines.append(
                f"{str(row.get('Start','')):<{col_w[0]}} "
                f"{str(row.get('End','')):<{col_w[1]}} "
                f"{str(row.get('Speaker','')):<{col_w[2]}} "
                f"{str(row.get('Text',''))}"
            )
        with open(dest, 'w', encoding='utf-8-sig') as f:
            f.write("\n".join(lines))
        self.log(f" Saved → {dest}")

    def _save_csv(self):
        if not self._rows:
            QMessageBox.warning(self, "No data", "Run the pipeline first.")
            return
        from datetime import datetime
        stem     = os.path.splitext(os.path.basename(self.lineEdit_path.text()))[0]
        default  = os.path.join(RESULT_DIR, f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", default, "CSV Files (*.csv)"
        )
        if dest:
            import pandas as pd
            pd.DataFrame(self._rows).to_csv(dest, index=False, encoding='utf-8-sig')
            self.log(f" Saved → {dest}")

    # ── Close ────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
        self._cleanup_thread()
        event.accept()
        QApplication.quit()

    def _set_save_enabled(self, on):
        self.pushButton_txt.setEnabled(on)
        self.pushButton_csv.setEnabled(on)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback, logging

    os.environ['MPLBACKEND'] = 'Agg'

    # ── Write all errors to a log file next to the exe ────────────────
    _log_path = os.path.join(os.path.dirname(sys.executable)
                             if getattr(sys, 'frozen', False)
                             else os.path.dirname(os.path.abspath(__file__)),
                             "app_crash.log")
    logging.basicConfig(
        filename=_log_path, filemode='a',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.WARNING, encoding='utf-8',
    )
    # Only log our app's INFO messages, not numba/librosa debug spam
    _app_handler = logging.FileHandler(_log_path, mode='a', encoding='utf-8')
    _app_handler.setLevel(logging.INFO)
    _app_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger(__name__).addHandler(_app_handler)
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.info("=== App started ===")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # Catch any unhandled exception — write to log + show popup
    def _except_hook(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.critical("UNHANDLED EXCEPTION:\n%s", msg)
        print("[EXCEPTION]", msg)
        try: QMessageBox.critical(None, "Error", msg[:800])
        except Exception: pass
    sys.excepthook = _except_hook

    try:
        window = MainWindow()
    except Exception:
        tb = traceback.format_exc()
        logging.critical("STARTUP CRASH:\n%s", tb)
        print("[STARTUP CRASH]", tb)
        QMessageBox.critical(None, "Startup Error", tb[:800])
        sys.exit(1)

    window.show()
    window.raise_()
    window.activateWindow()

    logging.info("=== App exiting ===")
    sys.exit(app.exec())