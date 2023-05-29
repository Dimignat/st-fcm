import numpy as np
import pandas as pd

from typing import Any, Union, Tuple
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV


class SemiSupervisedModel:
    """
    A self-training algorithm based on Fuzzy C-Means clustering
    """
    def __init__(self, model: Any, clustering: Any, dataloader, kernel_approx=None):
        """

        :param model: Any classification model available in the scikit-learn package.
        The model must have the fit, predict, and either predict_proba or decision_function methods.
        :param clustering: An algorithm for membership estimation, implementing the fit and soft_predict methods.
        :param dataloader: A dataloader inherited from the AbstractDataLoader class.
        :param kernel_approx (optional): A kernel approximation model from the scikit-learn package.
        """
        self.model = model
        self.clustering = clustering
        self.kernel_approx = kernel_approx
        self.dl = dataloader

        self.labeled = None
        self.labels = None
        self.unlabeled = None
        self.unique_labels = self.dl.get_unique_labels()
        self.verbose = None

    def __transform_kernel_approx(self, x) -> NDArray:
        """
        Transform data using kernel approximation.
        :param x: features data
        :return: transformed features data
        """
        if self.kernel_approx is not None:
            return self.kernel_approx.transform(x)
        return x

    def __fit_transform_kernel_approx(self, x, y=None) -> NDArray:
        """
        Transform data using kernel approximation.
        :param x: features data
        :param y (optional): labels data
        :return: transformed features data
        """
        if self.kernel_approx is not None:
            return self.kernel_approx.fit_transform(x, y)
        return x

    def __get_new_sets(self, tau: float, tau_step: float, t: float) -> \
            Union[Tuple[pd.DataFrame, NDArray], Tuple[None, None]]:
        """
        Acquire new labeled observations from the unlabeled set.
        :param tau: clustering threshold
        :param tau_step: clustering threshold decrease step
        :param t: model certainty threshold
        :return: new labeled sets
        """
        self.clustering.fit(self.labeled.to_numpy(), self.labels)
        unlabeled_probs = self.clustering.soft_predict(self.unlabeled.to_numpy())
        candidates = self.unlabeled[unlabeled_probs > t]
        try:
            transformed_candidates = self.__transform_kernel_approx(candidates)
            candidates_labels = self.model.predict(transformed_candidates)
        except ValueError as e:
            self.__log(f'Prediction error: {e}. Stopping.')
            return None, None

        try:
            calib_model = CalibratedClassifierCV(self.model, method='isotonic', cv='prefit', n_jobs=-1)
            calib_model.fit(transformed_candidates, candidates_labels)
            candidates_proba = calib_model.predict_proba(transformed_candidates)
        except Exception as e:
            self.__log(f'Calibration error: {e}. Stopping.')
            return None, None

        curr_tau = tau
        while curr_tau >= 0.5:
            condition = (np.argmax(candidates_proba, axis=1) == candidates_labels) & \
                        (np.amax(candidates_proba, axis=1) > curr_tau)
            new_labeled, new_labels = candidates[condition], candidates_labels[condition]
            if len(new_labels) == 0:
                curr_tau -= tau_step
            else:
                return new_labeled, new_labels

        self.__log('Cannot find more samples to label. Stopping.')
        return None, None

    def __set_datasets(self) -> None:
        """
        Load datasets from the dataloader.
        """
        if self.labeled is None or self.labels is None:
            self.labeled, self.labels = self.dl.get_labeled()
        if self.unlabeled is None:
            self.unlabeled = self.dl.get_unlabeled()

    def __log(self, msg) -> None:
        """
        Log messages.
        :param msg: A message to log
        """
        if self.verbose:
            print(msg)

    def increase_labeled(self, t=0.8, tau=0.8, tau_step=0.2, tol=0.001, min_perc=0.2, verbose=True) -> None:
        """
        A general method for increasing the labeled dataset with observations from the unlabeled dataset
        :param t: model certainty threshold
        :param tau: clustering threshold
        :param tau_step: clustering threshold decrease step
        :param tol: tolerance threshold (early stopping)
        :param min_perc: the minimum percentage of unlabeled observations to be left for the algorithm to stop.
        :param verbose: determines whether the algorithm steps are logged.
        """
        self.verbose = verbose

        self.__set_datasets()
        initial_size = self.unlabeled.shape[0]

        step = 0
        unlabeled_ratio = 1.0
        while True:
            step += 1

            labeled = self.__fit_transform_kernel_approx(self.labeled, self.labels)
            self.model.fit(labeled, self.labels)
            new_labeled, new_labels = self.__get_new_sets(tau, tau_step, t)
            if new_labels is None:
                break

            self.labeled = pd.DataFrame(np.concatenate((self.labeled, new_labeled), axis=0))
            self.labels = np.concatenate((self.labels, new_labels))
            self.unlabeled = self.unlabeled.drop(new_labeled.index).reset_index(drop=True)

            self.__log(f'Step #{step}: unlabeled: {self.unlabeled.shape[0] / initial_size * 100 :.2f}%')

            if unlabeled_ratio - self.unlabeled.shape[0] / initial_size < tol:
                self.__log('Tolerance stop.')
                break
            if self.unlabeled.shape[0] / initial_size < min_perc:
                self.__log('Min percentage stop.')
                break

            unlabeled_ratio = self.unlabeled.shape[0] / initial_size

            if self.unlabeled.shape[0] == 0:
                break

    def fit(self):
        """
        Fits the model on the labeled set.
        """
        self.__set_datasets()
        labeled = self.__fit_transform_kernel_approx(self.labeled, self.labels)
        self.model.fit(labeled, self.labels)

    def predict(self, x) -> NDArray:
        """
        Acquire predictions from the model
        :param x: features set to predict
        :return: predictions
        """
        x = self.__transform_kernel_approx(x)
        return self.model.predict(x)
