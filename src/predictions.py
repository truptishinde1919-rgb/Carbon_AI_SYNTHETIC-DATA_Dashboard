"""
predictions.py
--------------
Turns the user's process inputs into predicted carbon / electrical values.
"""
import pandas as pd
from src.preprocessing import FEATURES, LABELS


def out_of_range_warnings(inputs: dict, bundle: dict) -> list:
    """List inputs that lie outside the range seen in the dataset (extrapolation)."""
    notes = []
    for col, (lo, hi) in bundle["feature_ranges"].items():
        if not lo <= inputs[col] <= hi:
            notes.append(f"{LABELS[col]} = {inputs[col]:g} is outside the dataset range "
                         f"({lo:g} – {hi:g}); the prediction is an extrapolation.")
    return notes


def _predict(bundle, row: pd.DataFrame) -> float:
    return max(0.0, float(bundle["pipeline"].predict(row)[0]))  # physical values are never negative


def predict_all(main: dict, aux: dict, inputs: dict) -> dict:
    """Return every predicted value for one set of process inputs."""
    row = pd.DataFrame([inputs])[FEATURES]
    res = {
        "yield_pct": min(100.0, _predict(main["carbon_yield"], row)),
        "capacitance": _predict(main["capacitance"], row),
        "power_model": _predict(main["power"], row),
        "energy_model": _predict(main["energy"], row),
    }
    for key, b in aux.items():
        res[key] = _predict(b, row)

    # Transparent calculations (formulas shown on the dashboard)
    res["carbon_mass"] = inputs["Raw_Material_Mass_g"] * res["yield_pct"] / 100
    res["power_formula"] = res["voltage"] * res["current"]                        # P = V x I
    res["energy_formula"] = 0.5 * res["capacitance"] * res["voltage"] ** 2 / 3600  # simplified
    return res
