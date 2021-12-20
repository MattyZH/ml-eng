from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
from script.structures import train_test_split, get_precision_recall_accuracy, KDTree
from scipy.stats import bernoulli


class TestTTS:
    def test_true_shaping(self):
        x = np.zeros(100)
        y = np.zeros(100)
        for ratio in np.arange(0.1, 1, 0.1):
            x_train, y_train, x_test, y_test = train_test_split(x, y, ratio)
            assert np.allclose(x_train.shape[0], 100 * ratio)
            assert np.allclose(y_train.shape[0], 100 * ratio)
            assert np.allclose(x_test.shape[0], 100 * (1 - ratio))
            assert np.allclose(y_test.shape[0], 100 * (1 - ratio))

    def test_true_splitting(self):
        for ratio in np.arange(0.01, 1, 0.01):
            x = np.random.randn(100)
            y = np.random.randn(100)
            x_train, y_train, x_test, y_test = train_test_split(x, y, ratio)
            assert np.allclose(np.sort(np.concatenate((x_train, x_test))), np.sort(x))
            assert np.allclose(np.sort(np.concatenate((y_train, y_test))), np.sort(y))

    def test_corresponding_splitting(self):
        pass


class TestGPRA:
    def test_precs(self):
        for _ in range(10):
            y_pred = bernoulli.rvs(p=1/2, size=100)
            y_true = bernoulli.rvs(p=1 / 2, size=100)
            assert np.allclose(get_precision_recall_accuracy(y_pred, y_true)[0],
                               precision_recall_fscore_support(y_true, y_pred)[0])

    def test_recs(self):
        for _ in range(10):
            y_pred = bernoulli.rvs(p=1 / 2, size=100)
            y_true = bernoulli.rvs(p=1 / 2, size=100)
            assert np.allclose(get_precision_recall_accuracy(y_pred, y_true)[1],
                               precision_recall_fscore_support(y_true, y_pred)[1])

    def test_accs(self):
        for _ in range(10):
            y_pred = bernoulli.rvs(p=1 / 2, size=100)
            y_true = bernoulli.rvs(p=1 / 2, size=100)
            assert get_precision_recall_accuracy(y_pred, y_true)[2] == accuracy_score(y_true, y_pred)


class TestTree:
    def test_query(self):
        def true_closest(x_train, x_test, k):
            result = []
            for x0 in x_test:
                bests = list(sorted([(i, np.linalg.norm(x - x0)) for i, x in enumerate(x_train)], key=lambda x: x[1]))
                bests = [i for i, d in bests]
                result.append(bests[:min(k, len(bests))])
            return result

        for _ in range(100):
            x_train = np.random.randn(100, 3)
            x_test = np.random.randn(10, 3)
            tree = KDTree(x_train, leaf_size=2)
            predicted = tree.query(x_test, k=3)
            true = true_closest(x_train, x_test, k=3)
            assert np.sum(np.abs(np.array(np.array(predicted).shape) == np.array(np.array(true).shape)))
