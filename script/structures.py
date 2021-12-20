import numpy as np
import random
from typing import NoReturn, Tuple, List


def train_test_split(X: np.array, y: np.array, ratio: float
                     ) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    _size = int(len(y) * ratio)
    _indices_shuffle = random.sample(list(range(len(y))), k=len(y))
    _train = _indices_shuffle[:_size]
    _test = _indices_shuffle[_size:]

    return X[_train], y[_train], X[_test], y[_test]


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array
                                  ) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    _classes = np.unique(y_true)
    recs, precs = np.zeros(_classes.shape[0]), np.zeros(_classes.shape[0])
    for i, cls in enumerate(_classes):
        cls_true = (y_true == cls).astype(np.int)
        cls_pred = (y_pred == cls).astype(np.int)

        tp = np.sum(cls_pred[cls_true == 1] == 1)
        fp = np.sum(cls_pred[cls_true == 0] == 1)
        fn = np.sum(cls_pred[cls_true == 1] == 0)

        recs[i] = tp / (tp + fn)
        precs[i] = tp / (tp + fp)

    accuracy = np.sum(y_pred == y_true) / y_pred.shape[0]

    return precs, recs, accuracy


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.X = X
        self.leaf_size = leaf_size
        dim = len(X[0])
        self.dim = dim

        class Node:
            def __init__(self, population: list, parent=None, coord: int = -1, val: float = 0):
                """
                population:
                    list of indices of points in the node
                parent:
                    parent node
                coord:
                    coordinate of splitting with sibling node
                val:
                    value of splitting with sibling node
                """
                self.population = population
                self.parent = parent
                self.coord = coord
                self.val = val
                self.child = None

        init_node = Node(population=[i for i in range(len(X))])
        self.init_node = init_node
        self.node_que = [init_node]
        for node in self.node_que:
            _pop = node.population
            if len(_pop) > leaf_size:
                _pop_dots = X[_pop]
                c = (node.coord + 1) % dim  # new coordinate for splitting
                pivot = np.median(_pop_dots[:, c])  # pivot of splitting
                pop_1 = []
                pop_2 = []
                for i in _pop:
                    if X[i][c] > pivot:
                        pop_1.append(i)
                    else:
                        pop_2.append(i)
                node_1 = Node(population=pop_1, parent=node, coord=c, val=pivot)
                node_2 = Node(population=pop_2, parent=node, coord=c, val=pivot)
                node.child = (node_1, node_2)
                self.node_que.extend([node_1, node_2])

    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """

        def true_close(x0: np.array, inds: list, n: int = 1):
            bests = list(sorted([(i, np.linalg.norm(x - x0)) for i, x in enumerate(self.X[inds])], key=lambda x: x[1]))
            return bests[:min(n, len(bests))]

        ans = []
        for dot in X:
            leaf = self.init_node
            while leaf.child and len(leaf.population) > 2 * k:
                if dot[leaf.coord] > leaf.val:
                    leaf = leaf.child[0]
                else:
                    leaf = leaf.child[1]
            node = leaf

            closest = true_close(x0=dot, inds=node.population, n=k)
            max_of_closest = closest[-1][1]
            node_1 = node
            dist = [abs(dot[node_1.coord] - node_1.val)]
            while node_1.parent:
                node_1 = node_1.parent
                dist.append(abs(dot[node_1.coord] - node_1.val))
            height = len(dist)
            upper = 0
            for i in range(2, height):
                if dist[height - i] <= max_of_closest:
                    upper = height - i
                    break
            for i in range(upper + 1):
                node = node.parent
            closest = true_close(x0=dot, inds=node.population, n=k)

            ans.append([i[0] for i in closest])
        return ans


class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """

        self.mins = []
        self.maxs = []
        for i in range(len(X[0])):
            self.mins.append(min(X[:, i]))
            self.maxs.append(max(X[:, i]))
            X[:, i] = np.array((X[:, i] - self.mins[i]) * 1.0 / (self.maxs[i] - self.mins[i]))

        self.tree = KDTree(X=X, leaf_size=self.leaf_size)
        self.y = y
        self.classes = len(np.unique(y))

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """

        for i in range(X.shape[1]):
            X[:, i] = np.array((X[:, i] - self.mins[i]) * 1.0 / (self.maxs[i] - self.mins[i]))
        neighbor_list = self.tree.query(X, k=self.n_neighbors)
        ans = []
        n_classes = len(set(self.y))
        for neighbors in neighbor_list:
            css = [0] * n_classes
            for i in neighbors:
                css[self.y[i]] += 1
            ans.append(np.array([css[i] * 1.0 / self.n_neighbors for i in range(n_classes)]))
        return ans

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)
