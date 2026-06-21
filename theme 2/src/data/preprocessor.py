import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Optional, List


class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        df_copy = df.copy()
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns
        
        # Skip datetime columns
        numeric_cols = [col for col in numeric_cols if col not in datetime_cols]
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        
        if strategy == "mean":
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].mean())
        elif strategy == "median":
            df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median())
        elif strategy == "mode":
            for col in numeric_cols:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else 0)
            for col in categorical_cols:
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
                else:
                    # If all values are NaN, fill with a placeholder
                    df_copy[col] = df_copy[col].fillna("N/A")
        elif strategy == "drop":
            df_copy = df_copy.dropna()
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = "label") -> pd.DataFrame:
        df_copy = df.copy()
        for col in columns:
            if method == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                df_copy[col] = self.encoders[col].fit_transform(df_copy[col].astype(str))
            elif method == "onehot":
                df_copy = pd.get_dummies(df_copy, columns=[col], drop_first=True)
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame:
        df_copy = df.copy()
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        self.scalers['_'.join(columns)] = scaler
        return df_copy
    
    def parse_dates(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        df_copy = df.copy()
        for col in date_columns:
            df_copy[col] = pd.to_datetime(df_copy[col])
        return df_copy
