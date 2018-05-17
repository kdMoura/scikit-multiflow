import os
import numpy as np
from skmultiflow.data.generators.stagger_generator import STAGGERGenerator


def test_stagger_generator(test_path):
    stream = STAGGERGenerator(classification_function=2, sample_seed=112, balance_classes=False)
    stream.prepare_for_use()

    assert stream.n_remaining_samples() == -1

    expected_names = ["size", "color", "shape"]
    assert stream.get_feature_names() == expected_names

    expected_targets = [0, 1]
    assert stream.get_targets() == expected_targets

    assert stream.get_target_names() == ['class']

    assert stream.get_n_features() == 3

    assert stream.get_n_cat_features() == 0

    assert stream.get_n_num_features() == 3

    assert stream.get_n_targets() == 1

    assert stream.get_name() == 'Sine Generator - 1 target, 2 classes'

    assert stream.has_more_samples() is True

    assert stream.is_restartable() is True


    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'stagger_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    X, y = stream.last_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    stream.restart()
    X, y = stream.next_sample(10)
    assert np.alltrue(X == X_expected)
    assert np.alltrue(y == y_expected)

    assert stream.get_n_targets() == np.array(y).ndim

    assert stream.get_n_features() == X.shape[1]