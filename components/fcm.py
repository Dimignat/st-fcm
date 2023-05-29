import numpy as np
import pandas as pd

from numpy.typing import NDArray
from typing import Optional, Dict, Union, Callable
from enum import Enum
from joblib import Parallel, delayed


class DistanceOptions(str, Enum):
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    cosine = 'cosine'


class FCM:
    """
    A modifier version of the Fuzzy C-Means algorithm.
    Reference: https://github.com/omadson/fuzzy-c-means
    """
    def __init__(self, m=2.0, random_state=42, n_jobs=1):
        """
        Initialize the model
        :param m: fuzziness parameter
        :param random_state: random seed
        :param n_jobs: the number of CPUs to use
        """
        self.m = m
        self._centers = None
        self.u = None

        self.random_state: int = random_state
        self.rng = np.random.default_rng(self.random_state)
        self.n_jobs: int = n_jobs

        self.distance: Optional[Union[DistanceOptions, Callable]] = DistanceOptions.euclidean
        self.distance_params: Optional[Dict] = {}

    @staticmethod
    def _get_u(y: NDArray) -> NDArray:
        """
        Get the membership matrix for the initial dataset.
        :param y: the labels vector
        :return: membership matrix
        """
        y_series = pd.Series(y)
        n_classes = y_series.nunique()
        res = []
        for class_ in range(n_classes):
            class_size = y[y == class_].shape[0]
            tmp = np.zeros((class_size, n_classes))
            tmp[:, class_] = 1
            res.append(tmp)
        return np.concatenate(res, axis=0)

    @staticmethod
    def _sort_x(x: NDArray, y: NDArray):
        """
        Sort features and labels
        :param x: features
        :param y: labels
        :return: sorted features and labels
        """
        x_frame = pd.DataFrame(x)
        x_frame['label'] = y
        return x_frame.sort_values('label', ascending=True).drop('label', axis=1).to_numpy()

    def fit(self, x: NDArray, y: NDArray) -> None:
        """
        Fits the FCM algorithm
        :param x: features
        :param y: labels
        """
        x = FCM._sort_x(x, y)
        self.u = FCM._get_u(y)
        self._centers = FCM._next_centers(x, self.u, self.m)
        self.u = self.soft_predict(x)

    def soft_predict(self, x: NDArray) -> NDArray:
        """
        Predicts the membership values for the input features
        :param x: features
        :return: membership matrix
        """
        temp = FCM._dist(x, self._centers, self.distance, self.distance_params) ** (2 / (self.m - 1))
        u_dist = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda data, col: (data[:, col] / data.T).sum(0))(temp, col)
            for col in range(temp.shape[1])
        )
        u_dist = np.vstack(u_dist).T
        return 1 / u_dist

    @staticmethod
    def _dist(a: NDArray, b: NDArray, distance: Union[str, Callable], distance_params: Dict) -> NDArray:
        """
        Calculate distance between two matrices
        :param a: array 1
        :param b: array 2
        :param distance: distance metric
        :param distance_params: additional distance metric parameters
        :return: distance matrix
        """
        if isinstance(distance, Callable):
            return distance(a, b, distance_params)
        elif distance == 'minkowski':
            return FCM._minkowski(a, b, distance_params.get("p", 1.0))
        elif distance == 'cosine':
            return FCM._cosine_similarity(a, b)
        else:
            return FCM._euclidean(a, b)

    @staticmethod
    def _euclidean(a: NDArray, b: NDArray) -> NDArray:
        """
        Calculate Euclidian distance
        :param a: array 1
        :param b: array 2
        :return: distance
        """
        return np.sqrt(np.einsum("ijk->ij", (a[:, None, :] - b) ** 2))

    @staticmethod
    def _minkowski(a: NDArray, b: NDArray, p: float) -> NDArray:
        """
        Calculate Minkowski distance
        :param a: array 1
        :param b: array 2
        :return: distance
        """
        return (np.einsum("ijk->ij", (a[:, None, :] - b) ** p)) ** (1 / p)

    @staticmethod
    def _cosine_similarity(a: NDArray, b: NDArray) -> NDArray:
        """
        Calculate cosine similarity
        :param a: array 1
        :param b: array 2
        :return: distance
        """
        p1 = np.sqrt(np.sum(a ** 2, axis=1))[:, np.newaxis]
        p2 = np.sqrt(np.sum(b ** 2, axis=1))[np.newaxis, :]
        return np.dot(a, b.T) / (p1 * p2)

    @staticmethod
    def _next_centers(x: NDArray, u: NDArray, m: float) -> NDArray:
        """
        Calculate centroids
        :param x:
        :param u: membership matrix
        :param m: fuzziness parameter
        :return: centroids matrix
        """
        um = u ** m
        return (x.T @ um / np.sum(um, axis=0)).T
