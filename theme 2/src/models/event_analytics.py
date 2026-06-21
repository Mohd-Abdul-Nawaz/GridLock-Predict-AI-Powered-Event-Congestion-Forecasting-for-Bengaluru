import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import joblib


class EventCongestionForecaster:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame, event_cols: List[str], 
                        time_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df_copy = df.copy()
        
        if time_cols:
            for col in time_cols:
                df_copy[f"{col}_hour"] = df_copy[col].dt.hour
                df_copy[f"{col}_dayofweek"] = df_copy[col].dt.dayofweek
                df_copy[f"{col}_month"] = df_copy[col].dt.month
        
        self.feature_columns = [col for col in df_copy.columns if col not in ['congestion_level', 'event_id']]
        return df_copy
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(n_estimators=100, random_state=random_state, objective="reg:squarederror")
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1)
        
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {"train_r2": train_score, "test_r2": test_score}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def recommend_resources(self, predicted_congestion: float, event_type: str) -> Dict:
        if predicted_congestion > 0.8:
            manpower = 20
            barricades = 15
            diversions = 3
        elif predicted_congestion > 0.5:
            manpower = 12
            barricades = 8
            diversions = 2
        else:
            manpower = 6
            barricades = 4
            diversions = 1
        
        return {
            "predicted_congestion": predicted_congestion,
            "recommended_manpower": manpower,
            "recommended_barricades": barricades,
            "recommended_diversions": diversions,
            "priority": "Critical" if predicted_congestion > 0.8 else "High" if predicted_congestion > 0.5 else "Medium"
        }
    
    def save_model(self, filepath: str):
        joblib.dump({"model": self.model, "feature_columns": self.feature_columns, "model_type": self.model_type}, filepath)
    
    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.model_type = data["model_type"]
