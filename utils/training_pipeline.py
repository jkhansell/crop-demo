import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_state: int = 123
    
class CropPipeline:
    """
    Minimal "experiment runner" for:
      - classification: predict label
      - regression: predict yield

    You can swap models by passing different estimators.
    """
    def __init__(
        self,
        cfg: TrainConfig,
        clf_model=None,
        scale_numeric: bool = True,
    ):
        self.cfg = cfg
        self.scale_numeric = scale_numeric

        assert clf_model != None, "Please pass a Classifier"

        self.clf_model = clf_model
        
        self.clf_pipe: Optional[Pipeline] = None

        self.results = {
            "clf_accuracy": None,
            "clf_f1_macro": None,
        }


    def _make_preprocessor(self, num_cols):
        if self.scale_numeric:
            return ColumnTransformer(
                transformers=[("num", StandardScaler(), num_cols)],
                remainder="drop",
            )
            
        # no scaling (useful for tree models)
        return ColumnTransformer(
            transformers=[("num", "passthrough", num_cols)],
            remainder="drop",
        )

    def fit_classification(self, X: pd.DataFrame, y: pd.Series, num_cols) -> Dict[str, Any]:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state, stratify=y
        )

        pre = self._make_preprocessor(num_cols)
        self.clf_pipe = Pipeline([("pre", pre), ("model", self.clf_model)])
        self.clf_pipe.fit(X_tr, y_tr)

        pred = self.clf_pipe.predict(X_te)

        self.results["clf_accuracy"] = float(accuracy_score(y_te, pred))
        self.results["clf_f1_macro"] = float(f1_score(y_te, pred, average="macro"))
        
    def predict_label(self, X: pd.DataFrame) -> np.ndarray:
        if self.clf_pipe is None:
            raise RuntimeError("Classification pipeline not fit yet.")
        return self.clf_pipe.predict(X)