import sys, types

# Fix debugpy conflict — force-load sklearn compiled modules before debugpy intercepts
try:
    import sklearn.pipeline, sklearn.preprocessing
    import sklearn.ensemble, sklearn.ensemble._forest
    import sklearn.tree, sklearn.tree._classes
    import sklearn.svm, sklearn.neighbors, sklearn.dummy
    import sklearn.metrics, sklearn.model_selection
    import sklearn.utils, sklearn.base
    import scipy.special, scipy.linalg, scipy.sparse
except ImportError:
    pass

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pyannote.core import Segment, Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


class SpeakerClassifier:
    def __init__(self, n_splits=5):
        self.n_splits        = n_splits
        self.all_labels      = []
        self.is_trained      = False
        self.best_model_name = 'RF'
        self.pipelines = {
            'RF': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    RandomForestClassifier(
                               n_estimators=300,       # more trees = more stable
                               max_depth=None,
                               min_samples_leaf=2,     # avoid overfitting tiny nodes
                               # handles speaker imbalance
                               random_state=42,
                               n_jobs=-1))             # use all CPU cores
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    SVC(kernel='rbf',           # rbf often better than linear for audio
                               C=10,
                               gamma='scale',
                               probability=True,
                               
                               random_state=42))
            ]),
            'KNN': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    KNeighborsClassifier(
                               n_neighbors=7,          # slightly more neighbours = smoother
                               weights='distance',
                               metric='euclidean',
                               n_jobs=-1))
            ]),
            'GBM': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    GradientBoostingClassifier(
                               n_estimators=150,
                               learning_rate=0.1,
                               max_depth=4,
                               random_state=42))
            ]),
            'Baseline (Majority)': Pipeline([
                ('scaler', StandardScaler()),
                ('clf',    DummyClassifier(strategy='most_frequent', random_state=42))
            ]),
        }
        self._inference_pipeline = None

    def train(self, X, y):
        self.all_labels = sorted(set(y))
        X = np.array(X)
        y = np.array(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        print(f"\n  Stratified {self.n_splits}-Fold CV | "
              f"{len(X)} segments | {len(self.all_labels)} speakers: {self.all_labels}")
        print(f"  Feature vector: {X.shape[1]}-dim (MFCC + delta + delta² + std)")

        results      = {}
        best_acc     = -1
        best_name    = 'RF'

        for name, pipeline in self.pipelines.items():
            fold_scores, y_true_all, y_pred_all = [], [], []
            for train_idx, test_idx in skf.split(X, y):
                pipeline.fit(X[train_idx], y[train_idx])
                y_pred = pipeline.predict(X[test_idx])
                fold_scores.append(accuracy_score(y[test_idx], y_pred))
                y_true_all.extend(y[test_idx].tolist())
                y_pred_all.extend(y_pred.tolist())

            cv_acc   = float(np.mean(fold_scores))
            cv_std   = float(np.std(fold_scores))
            report   = classification_report(
                y_true_all, y_pred_all, labels=self.all_labels,
                output_dict=True, zero_division=0)
            macro_f1 = report['macro avg']['f1-score']

            results[name] = {
                'cv_acc':   cv_acc,
                'cv_std':   cv_std,
                'cv_scores': fold_scores,
                'macro_f1': macro_f1,
                'report':   report,
                'conf':     confusion_matrix(y_true_all, y_pred_all,
                                             labels=self.all_labels),
            }

            is_base = 'Baseline' in name
            marker  = ' ← baseline' if is_base else ''
            print(f"  {name:<28} CV Acc: {cv_acc:.2%} ± {cv_std:.2%}"
                  f"  Macro F1: {macro_f1:.2%}{marker}")

            # Auto-select best non-baseline model by macro F1
            if not is_base and macro_f1 > best_acc:
                best_acc  = macro_f1
                best_name = name

        # Refit best model on ALL data for inference
        self.best_model_name     = best_name
        self._inference_pipeline = self.pipelines[best_name]
        self._inference_pipeline.fit(X, y)
        self.is_trained = True

        print(f"\n  ★ Best model: {best_name} "
              f"(CV Acc={results[best_name]['cv_acc']:.2%}, "
              f"Macro F1={results[best_name]['macro_f1']:.2%}) "
              f"— refit on all {len(X)} segments.")
        return results

    @staticmethod
    def compute_der(reference_segments, hypothesis_segments):
        if not reference_segments:
            return 0.0
        if not hypothesis_segments:
            print("  ⚠️  No hypothesis segments — DER cannot be computed.")
            return 1.0
        reference  = Annotation()
        hypothesis = Annotation()
        for seg in reference_segments:
            reference[Segment(seg['start'], seg['end'])]  = str(seg['speaker'])
        for seg in hypothesis_segments:
            hypothesis[Segment(seg['start'], seg['end'])] = str(seg['speaker'])
        uem = Timeline(segments=[
            Segment(seg['start'], seg['end']) for seg in reference_segments
        ])
        return float(DiarizationErrorRate()(reference, hypothesis, uem=uem))

    def predict(self, feature):
        if not self.is_trained:
            raise RuntimeError("Call train() first.")
        return str(self._inference_pipeline.predict([feature])[0])

    def predict_proba(self, feature):
        """Return confidence scores for each speaker. Requires probability=True in pipeline."""
        if not self.is_trained:
            raise RuntimeError("Call train() first.")
        clf = self._inference_pipeline.named_steps['clf']
        if not hasattr(clf, 'predict_proba'):
            return None
        scaler = self._inference_pipeline.named_steps['scaler']
        feat_scaled = scaler.transform([feature])
        proba = clf.predict_proba(feat_scaled)[0]
        return dict(zip(self.all_labels, proba))