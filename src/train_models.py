"""
train_models.py
---------------
Trains Random Forest and Linear Regression for every target, picks the
better one using 5-fold cross-validation R² on the TRAINING split, and
reports honest metrics on a hold-out TEST split (20 %).

All metrics are calculated from the dataset -- nothing is hard-coded.
Run manually with:   python src/train_models.py
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import (AUX_FILE, AUX_TARGETS, FEATURES, MAIN_TARGETS,
                               MODELS_DIR, NUMERIC_FEATURES, build_preprocessor,
                               load_data)

RANDOM_STATE = 42


def _make_models():
    """The two candidate algorithms (each wrapped with preprocessing)."""
    return {
        "Random Forest": Pipeline([
            ("prep", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                            random_state=RANDOM_STATE)),
        ]),
        "Linear Regression": Pipeline([
            ("prep", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
    }


def _metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _rf_importance(pipeline) -> pd.DataFrame:
    """Random-Forest importances; one-hot columns are merged back into 'Waste Material'."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    imp = pipeline.named_steps["model"].feature_importances_
    rows = {}
    for n, v in zip(names, imp):
        key = "Waste_Material" if n.startswith("cat__") else n.replace("num__", "")
        rows[key] = rows.get(key, 0.0) + float(v)
    return (pd.DataFrame({"Feature": list(rows), "Importance": list(rows.values())})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def train_target(df: pd.DataFrame, target_col: str) -> dict:
    """Train both models for ONE target and return a bundle with everything the app needs."""
    X, y = df[FEATURES], df[target_col]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    candidates, all_metrics = _make_models(), {}
    for name, pipe in candidates.items():
        cv_r2 = float(cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean())  # validation
        pipe.fit(X_tr, y_tr)
        all_metrics[name] = {"CV_R2": cv_r2, **_metrics(y_te, pipe.predict(X_te))}

    best = max(all_metrics, key=lambda n: all_metrics[n]["CV_R2"])  # choose by validation
    return {
        "target": target_col,
        "model_name": best,
        "pipeline": candidates[best],
        "metrics": all_metrics[best],
        "all_metrics": all_metrics,
        "y_test": y_te.to_numpy(),
        "y_pred": candidates[best].predict(X_te),
        "importance": _rf_importance(candidates["Random Forest"]),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "feature_ranges": {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC_FEATURES},
        "target_range": (float(y.min()), float(y.max())),
    }


def train_all(df: pd.DataFrame, models_dir=MODELS_DIR):
    """Train every model and save them to models/*.pkl. Returns (main, aux)."""
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    main = {}
    for key, (col, fname) in MAIN_TARGETS.items():
        main[key] = train_target(df, col)
        joblib.dump(main[key], Path(models_dir) / fname)
    aux = {key: train_target(df, col) for key, col in AUX_TARGETS.items()}
    joblib.dump(aux, Path(models_dir) / AUX_FILE)
    return main, aux


def load_models(models_dir=MODELS_DIR):
    """Load saved models; raises an error if any file is missing."""
    main = {k: joblib.load(Path(models_dir) / f) for k, (_, f) in MAIN_TARGETS.items()}
    aux = joblib.load(Path(models_dir) / AUX_FILE)
    return main, aux


if __name__ == "__main__":
    data = load_data()
    main_models, _ = train_all(data)
    for k, b in main_models.items():
        m = b["metrics"]
        print(f"{k:13s} best={b['model_name']:18s} R2={m['R2']:.3f} MAE={m['MAE']:.4f} RMSE={m['RMSE']:.4f}")
    print("Models saved to", MODELS_DIR)
