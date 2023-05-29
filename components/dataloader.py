import abc
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict, Tuple, Union
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder


class AbstractDataLoader(abc.ABC):
    """
    An abstract dataloader compatible with SemiSupervisedModel. All dataloader must be inherited from this one.
    """
    @abc.abstractmethod
    def get_labeled(self) -> Tuple[pd.DataFrame, NDArray]:
        ...

    @abc.abstractmethod
    def get_unlabeled(self) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def get_unique_labels(self) -> List[Union[str, int]]:
        ...


class QGISDataLoader(AbstractDataLoader):
    """
    A dataloader compatible with the "QGIS Modular Classification Toolkit"
    (https://github.com/Alex-Blade/qgis-classification-toolkit)
    """

    def __init__(self, labeled_batches: str, unlabeled_batches: str, random_state=42):
        """
        :param labeled_batches: Path to labeled batches.
        :param unlabeled_batches: Path to unlabeled batches.
        :param random_state: Random seed.
        """
        self.labeled_batches = [f for f in Path(labeled_batches).iterdir() if not f.name.startswith('.')]
        self.unlabeled_batches = [f for f in Path(unlabeled_batches).iterdir() if not f.name.startswith('.')]

        self.rng = np.random.default_rng(random_state)
        self.le = LabelEncoder()

    @staticmethod
    def _load_npz(npz_path: Path) -> NDArray:
        """
        Load data from a NPZ file.
        :param npz_path: The path to the NPZ file to load.
        :return: The loaded and transformed data as a NumPy array.
        """
        with open(npz_path, 'rb') as npz_file:
            npz = np.load(npz_file)
            arr = np.array([npz[f] for f in sorted(npz.files)])
        arr = arr.transpose(1, 2, 0)
        arr = arr.reshape(-1, 3)
        return arr[~np.all(arr == 0, axis=1)]

    def _extract_unlabeled(self) -> NDArray:
        """
        Extract unlabeled data.
        :return: Data as a NumPy array.
        """
        if len(self.unlabeled_batches) == 1:
            batch = self.unlabeled_batches[0]
            return self._load_npz(batch / 'unlabeled.npz')
        else:
            raise NotImplementedError

    def _extract_labeled(self) -> Dict[str, NDArray]:
        """
        Extract labeled data.
        :return: A dictionary with class names as keys and corresponding data arrays as values.
        """
        classes = {}

        for batch in self.labeled_batches:
            for npz_file in batch.iterdir():
                if npz_file.name.startswith('.'):
                    continue
                data = self._load_npz(npz_file)
                class_name = npz_file.name.rsplit('_')[0]
                if class_name in classes:
                    class_data = classes[class_name]
                    classes[class_name] = np.concatenate((class_data, data), axis=0)
                else:
                    classes[class_name] = data
        return classes

    def get_labeled(self) -> Tuple[pd.DataFrame, NDArray]:
        """
        Extracts labeled data and transforms it to be compatible with the SelfSupervisedModel.
        :return: A Pandas dataframe of features and corresponding labels array.
        """
        labeled_data = self._extract_labeled()

        labels, dataset = [], []
        for cluster, data in labeled_data.items():
            labels.append(np.array([cluster] * len(data)))
            dataset.append(data)
        labels, dataset = np.concatenate(labels), np.concatenate(dataset)
        labels_num = self.le.fit_transform(labels).reshape(-1)
        return pd.DataFrame(dataset), labels_num

    def get_unlabeled(self) -> pd.DataFrame:
        """
        Extracts unlabeled data and transforms it to be compatible with the SelfSupervisedModel.
        :return: A Pandas dataframe of features.
        """
        unlabeled_data = self._extract_unlabeled()
        return pd.DataFrame(unlabeled_data)

    def get_unique_labels(self) -> List[str]:
        """
        Returns the available label classes.
        :return: A list of class names.
        """
        class_names = []
        for batch in self.labeled_batches:
            for npz_file in batch.iterdir():
                if npz_file.name.startswith('.'):
                    continue
                class_name = npz_file.name.rsplit('_')[0]
                class_names.append(class_name)
        return list(set(class_names))
