"""Analyze confusion matrices at different SNRs to diagnose non-monotonic accuracy."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signal_processor import _get_processor
from dataset_builder import build_dataset, split_and_normalize
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from config import RANDOM_SEED

label_names = ['4ASK', '2FSK', 'BPSK', '8PSK']

for snr in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:
    print(f'\n=== SNR={snr:>2d}dB (700 train / 300 test per class) ===')
    X, y, _, _ = build_dataset(
        snr_db=snr, samples_per_class=1000,
        random_state=42, show_progress=False
    )
    X_train, X_test, y_train, y_test, _s = split_and_normalize(
        X, y, random_state=42
    )

    svm = SVC(kernel='linear', C=1.0, probability=False, random_state=RANDOM_SEED)
    model = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = np.trace(cm) / np.sum(cm)

    per_class = []
    for i, name in enumerate(label_names):
        ca = cm[i, i] / np.sum(cm[i, :]) * 100
        per_class.append(f"{name}={ca:.1f}%")

    print(f"  Overall: {acc:.4f}  |  {' | '.join(per_class)}")
    print(f"  Errors:  ", end="")
    errors = []
    for i in range(4):
        for j in range(4):
            if i != j and cm[i, j] > 0:
                errors.append(f"{label_names[i]}->{label_names[j]}:{cm[i,j]}")
    print(", ".join(errors) if errors else "none")

