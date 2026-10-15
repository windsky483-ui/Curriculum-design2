"""Regression checks for the final report's class mapping; no RadioML download needed."""
import contextlib
import io
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import MODULATION_TYPES, LEVEL1_GROUPS, LEVEL2_MODS
from ai_model import hierarchical_predict, load_per_snr_models, validate_model_labels
from dataset_builder import map_to_group_labels, filter_psk_samples
from dataset_loader import load_external_signals
from signal_processor import SignalProcessor
from signal_generator import generate_signal


class ReportModulationsTest(unittest.TestCase):
    def test_txt_and_pickle_load_report_classes(self):
        rng = np.random.RandomState(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_names = ['BPSK', 'QPSK', '8PSK', '16QAM', 'PAM4', 'GFSK']
            data = {}
            for name in original_names:
                iq = rng.normal(size=(2, 2, 128))
                data[(name, 0)] = iq
                signals = iq[:, 0] + 1j * iq[:, 1]
                (root / f'{name} 0.txt').write_text(
                    '\n'.join(' '.join(str(z) for z in sig) for sig in signals),
                    encoding='utf-8',
                )
            processor = SignalProcessor(directory)
            self.assertEqual(processor.get_available_mod_types(), MODULATION_TYPES)
            for name in MODULATION_TYPES:
                self.assertEqual(processor.get_all_signals(name, 0).shape, (2, 128))
            dataset_path = root / 'dataset.pkl'
            with dataset_path.open('wb') as stream:
                pickle.dump(data, stream)
            signals, labels, snrs = load_external_signals(str(dataset_path))
            self.assertEqual(len(signals), 8)
            self.assertEqual(set(labels), set(MODULATION_TYPES))
            self.assertEqual(snrs, [0] * 8)

    def test_hierarchy_labels_and_psk_filter(self):
        y = np.arange(4)
        X = np.arange(8).reshape(4, 2)
        groups, names = map_to_group_labels(y, MODULATION_TYPES)
        np.testing.assert_array_equal(groups, [0, 0, 0, 1])
        self.assertEqual(names, ['PSK', 'QAM'])
        X_psk, y_psk, labels = filter_psk_samples(X, y, MODULATION_TYPES)
        np.testing.assert_array_equal(X_psk, X[:3])
        np.testing.assert_array_equal(y_psk, [0, 1, 2])
        self.assertEqual(labels, ['BPSK', 'QPSK', '8PSK'])

    def test_trained_hierarchy_covers_all_four_outputs(self):
        rng = np.random.RandomState(12)
        centers = np.array([[-4, -4], [-4, 0], [-4, 4], [4, 0]])
        X = np.vstack([rng.normal(c, 0.15, size=(24, 2)) for c in centers])
        y = np.repeat(np.arange(4), 24)
        groups, _ = map_to_group_labels(y, MODULATION_TYPES)
        X_psk, y_psk, _ = filter_psk_samples(X, y, MODULATION_TYPES)
        s1, s2 = StandardScaler().fit(X), StandardScaler().fit(X_psk)
        m1 = SVC(probability=True, random_state=42).fit(s1.transform(X), groups)
        m2 = SVC(probability=True, random_state=42).fit(s2.transform(X_psk), y_psk)
        predictions, probabilities = hierarchical_predict(
            {'level1': m1, 'level2': m2}, {'level1': s1, 'level2': s2}, centers,
        )
        np.testing.assert_array_equal(predictions, [0, 1, 2, 3])
        self.assertEqual(probabilities.shape, (4, 4))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), predictions)

    def test_old_model_is_rejected(self):
        validate_model_labels(MODULATION_TYPES)
        validate_model_labels({'level1': LEVEL1_GROUPS, 'level2': LEVEL2_MODS})
        for old_labels in [
            ['4ASK', '2FSK', 'BPSK', '8PSK'],
            {'level1': ['ASK', 'FSK', 'PSK'], 'level2': ['BPSK', '8PSK']},
        ]:
            with self.assertRaisesRegex(ValueError, 'main.py --train'):
                validate_model_labels(old_labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.pkl'
            with path.open('wb') as stream:
                pickle.dump({'label_names': ['4ASK', '2FSK', 'BPSK', '8PSK']}, stream)
            with self.assertRaises(ValueError):
                load_per_snr_models(str(path))

    def test_generator_accepts_report_qam_name(self):
        for name in MODULATION_TYPES:
            signal, _, _ = generate_signal(name, snr_db=10, n_symbols=8)
            self.assertGreater(len(signal), 0)
            self.assertTrue(np.isfinite(signal).all())

    def test_main_training_with_small_fixture(self):
        import main
        rng = np.random.RandomState(8)
        centers = np.array([[-4, -4], [-4, 0], [-4, 4], [4, 0]])
        X = np.vstack([rng.normal(c, 0.2, size=(30, 2)) for c in centers])
        y = np.repeat(np.arange(4), 30)
        fixture = (X, y, ['f1', 'f2'], MODULATION_TYPES)
        with patch('database.init_database'), \
             patch('signal_processor.get_all_snr_levels', return_value=[0]), \
             patch('dataset_builder.build_dataset', return_value=fixture), \
             patch('ai_model.tune_hyperparams', return_value=(None, {'C': 5, 'gamma': 'scale'}, 0)), \
             patch('ai_model.save_per_snr_models') as save, \
             contextlib.redirect_stdout(io.StringIO()):
            models, scalers, _, labels, results = main.train_models()
        validate_model_labels(labels)
        save.assert_called_once()
        np.testing.assert_array_equal(models[0]['level2'].classes_, [0, 1, 2])
        self.assertEqual(len(results[0]['values']), 3)
        predictions, _ = hierarchical_predict(models[0], scalers[0], centers)
        np.testing.assert_array_equal(predictions, [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
