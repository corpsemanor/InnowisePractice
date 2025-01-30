import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from typing import List, Tuple, Optional
from sklearn.metrics import mean_squared_error

class ValidationTimeSeriesSplit:
    """
    Month-index based expanding window validation
    """
    def __init__(self, 
                 min_train_size: int = 12,
                 val_size: int = 1,       
                 gap: int = 0,            
                 test_size: int = 6):     
        
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.gap = gap
        self.test_size = test_size

    def split(self, df: pd.DataFrame) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.Index]:
        months = np.sort(df['date__month'].unique())
        max_month = months[-1]
        
        test_start_month = max_month - self.test_size - self.val_size + 1
        test_end_month = max_month
        
        current_train_end_month = self.min_train_size - 1
        splits = []
        
        while True:
            val_start_month = current_train_end_month + self.gap + 1
            val_end_month = val_start_month + self.val_size - 1
            
            if val_start_month > test_start_month or val_end_month > test_end_month:
                break
                
            train_mask = df['date__month'] <= current_train_end_month
            val_mask = (df['date__month'] >= val_start_month) & (df['date__month'] <= val_end_month)
            
            splits.append(
                (df[train_mask].index.to_numpy(), 
                 df[val_mask].index.to_numpy())
            )
            
            current_train_end_month = val_end_month

        test_mask = (df['date__month'] >= test_start_month) & (df['date__month'] <= test_end_month)
        test_idx = df[test_mask].index
        
        return splits, test_idx

    def validate_splits(self, df: pd.DataFrame, splits: list) -> bool:
        """
        Checks whether validation splits are correctly expanding and non-overlapping
        """
        prev_months = set()
        for train_idx, val_idx in splits:
            train_months = set(df.loc[train_idx, 'date__month'])
            val_size = set(df.loc[val_idx, 'date__month'])
            
            if not train_months.issuperset(prev_months):
                return False
            if train_months & val_size:
                return False
                
            prev_months = train_months
        return True

    def check_minimum_data(self, df: pd.DataFrame) -> bool:
        """
        Checks if there is enough data for at least one full validation split.
        """
        months = pd.to_datetime(df['date__month'].unique()).sort_values()
        required = self.min_train_size + self.gap + self.val_size
        return len(months) >= required + self.test_size

    def check_target_leakage(self, features: List[str], target: str, data: pd.DataFrame) -> bool:
        """
        Checks if any feature contains future information, which would leak into training.
        """
        for feature in features:
            if data[feature].shift(-1).isna().sum() < len(data) - self.min_train_size:
                return True
        if data[target].shift(-1).isna().sum() < len(data) - self.min_train_size:
            return True
        return False

    def check_validation_sufficiency(self, data: pd.DataFrame) -> Tuple[bool, pd.Series]:
        """
        Checks if the validation set is statistically similar to the test set.
        """
        test = data.iloc[self.split(data)[1]]
        val = data.iloc[self.split(data)[0][-1][1]]

        exclude_cols = {'date__month', 'date__month_of_year'}
        test = test.drop(columns=[col for col in exclude_cols if col in test.columns], errors='ignore')
        val = val.drop(columns=[col for col in exclude_cols if col in val.columns], errors='ignore')

        ks_test = np.abs(test.mean() - val.mean()) / (test.std() + 1e-9)
        return (ks_test < 0.1).all(), ks_test


    def check_data_adequacy(self, data: pd.DataFrame) -> bool:
        """
        Ensure enough validation samples for reliable model ranking
        """
        splits, _ = self.split(data)
        num_splits = len(splits)
        min_samples = 3 * self.val_size
        return len(data) >= (self.min_train_size + num_splits * self.val_size + self.gap)

class XGBModel:
    """
    XGBoost model with integrated expanding window training
    """
    def __init__(self, validator: ValidationTimeSeriesSplit, params: dict = None):
        self.validator = validator
        self.models = []
        self.params = params or {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 4,
            'objective': 'reg:squarederror',
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }

    def fit(self, data: pd.DataFrame, target_col: str = 'item_cnt_month'):
        """
        Train using expanding window strategy
        """
        splits, test_idx = self.validator.split(data)
        
        for i, (train_idx, val_idx) in enumerate(splits):
            X_train = data.iloc[train_idx].drop(columns=[target_col])
            y_train = data.iloc[train_idx][target_col]
            
            X_val = data.iloc[val_idx].drop(columns=[target_col])
            y_val = data.iloc[val_idx][target_col]

            model = XGBRegressor(**self.params)
            model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)
            
            self.models.append(model)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            print(f"Window {i+1} - Train: {train_idx[0]} to {train_idx[-1]} | "
                f"Val: {val_idx[0]} to {val_idx[-1]} | "
                f"Best Iteration: {model.best_iteration} | RMSE: {rmse:.4f}")

        return self

    def predict(self, X: pd.DataFrame, strategy: str = 'last') -> np.ndarray:
        """
        Predict using either:
        - 'last': Use final model from last window
        - 'ensemble': Average predictions from all models
        """
        if strategy == 'last':
            return self.models[-1].predict(X)
        elif strategy == 'ensemble':
            preds = np.zeros((len(X), len(self.models)))
            for i, model in enumerate(self.models):
                preds[:, i] = model.predict(X)
            return preds.mean(axis=1)
        else:
            raise ValueError("Invalid strategy. Use 'last' or 'ensemble'")

class FeatureExtractor():
    """
    TS feature extraction 
    """