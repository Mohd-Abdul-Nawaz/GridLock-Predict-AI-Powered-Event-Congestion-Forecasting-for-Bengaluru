import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath, **kwargs)
    
    def load_excel(self, filename: str, sheet_name: Optional[Union[str, int]] = 0, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    def load_json(self, filename: str, **kwargs) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_json(filepath, **kwargs)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = "csv") -> None:
        filepath = os.path.join(self.data_dir, filename)
        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "excel":
            df.to_excel(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records")
        else:
            raise ValueError("Unsupported format. Use 'csv', 'excel', or 'json'.")
