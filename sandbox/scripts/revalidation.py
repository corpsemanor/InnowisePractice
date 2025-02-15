import numpy as np
import pandas as pd

class ExpandingWindowSplitter:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None, *args, **kwargs):
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            n_samples = X.shape[0]
        else:
            raise ValueError("X must be a pandas DataFrame or numpy array")
        
        fold_size = n_samples // (self.n_splits + 1)
        if fold_size == 0:
            raise ValueError("Observation number is too low.")
        
        for i in range(self.n_splits):

            train_end = fold_size * (i + 1)
            test_end = train_end + fold_size# * (i + 2)

            if test_end > n_samples:
                test_end = n_samples
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(train_end, test_end)
            # print(f'{train_indices[0]}-{train_indices[-1]}|{test_indices[0]}-{test_indices[len(test_indices)-1]}')
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None, *args, **kwargs):
        return self.n_splits
