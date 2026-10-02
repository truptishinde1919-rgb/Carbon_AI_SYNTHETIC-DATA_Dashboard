"""
preprocessing.py
----------------
Everything related to LOADING and CLEANING the dataset, plus the
constants (column names, feature lists, targets) shared by the whole project.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "carbon_demo_dashboard_data.csv"
MODELS_DIR = BASE_DIR / "models"

# ---------------------------------------------------------------- columns
CATEGORICAL_FEATURES = ["Waste_Material"]
NUMERIC_FEATURES = [
    "Raw_Material_Mass_g",
    "Initial_Moisture_pct",
    "Carbonization_Temperature_C",
    "Carbonization_Time_min",
    "Activation_Ratio_KOH_or_Activator",
    "Activation_Temperature_C",
    "Activation_Time_min",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Main targets (one saved .pkl file each) -> (dataset column, file name)
MAIN_TARGETS = {
    "carbon_yield": ("Carbon_Yield_pct", "carbon_yield_model.pkl"),
    "capacitance": ("Capacitance_F_per_g", "capacitance_model.pkl"),
    "power": ("Output_Power_W", "power_model.pkl"),
    "energy": ("Energy_Storage_Wh", "energy_model.pkl"),
}
# Extra targets needed to show resistance / conductivity / voltage / current
AUX_TARGETS = {
    "resistance": "Resistance_Ohm",
    "conductivity": "Conductivity_S_per_cm",
    "voltage": "Voltage_V",
    "current": "Current_A",
}
AUX_FILE = "auxiliary_models.pkl"

REQUIRED_COLUMNS = (
    ["Sample_ID", "Test_Status", "Data_Type", "Carbon_Yield_pct", "Carbon_Mass_g",
     "Capacitance_F_per_g", "Resistance_Ohm", "Conductivity_S_per_cm",
     "Voltage_V", "Current_A", "Output_Power_W", "Energy_Storage_Wh"]
    + FEATURES
)

# Friendly labels used on charts and cards
LABELS = {
    "Waste_Material": "Waste Material",
    "Raw_Material_Mass_g": "Raw Material Mass (g)",
    "Initial_Moisture_pct": "Initial Moisture (%)",
    "Carbonization_Temperature_C": "Carbonization Temperature (°C)",
    "Carbonization_Time_min": "Carbonization Time (min)",
    "Activation_Ratio_KOH_or_Activator": "Activation Ratio",
    "Activation_Temperature_C": "Activation Temperature (°C)",
    "Activation_Time_min": "Activation Time (min)",
    "Carbon_Yield_pct": "Carbon Yield (%)",
    "Carbon_Mass_g": "Carbon Mass (g)",
    "Capacitance_F_per_g": "Capacitance (F/g)",
    "Resistance_Ohm": "Resistance (Ω)",
    "Conductivity_S_per_cm": "Conductivity (S/cm)",
    "Voltage_V": "Voltage (V)",
    "Current_A": "Current (A)",
    "Output_Power_W": "Output Power (W)",
    "Energy_Storage_Wh": "Energy Storage (Wh)",
}


class DataError(Exception):
    """Raised with a human-readable message when the dataset is unusable."""


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV, validate columns and handle missing / invalid values."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"Dataset not found at '{path}'. Place 'carbon_demo_dashboard_data.csv' "
            f"inside the 'data/' folder and restart the app."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # corrupted / empty file
        raise DataError(f"Could not read the CSV file: {exc}") from exc

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(f"The CSV is missing required columns: {missing_cols}")

    # Rows without an ID or material cannot be used
    df = df.dropna(subset=["Sample_ID", "Waste_Material"]).copy()

    # Force numeric columns to numbers (bad text becomes NaN), then fill
    # gaps with the column median so nothing crashes.
    numeric_cols = [c for c in REQUIRED_COLUMNS
                    if c not in ("Sample_ID", "Waste_Material", "Test_Status", "Data_Type")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    n_filled = int(df[numeric_cols].isna().sum().sum())
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df["Test_Status"] = df["Test_Status"].fillna("Unknown")

    if len(df) < 20:
        raise DataError("Too few valid rows (<20) to train reliable demo models.")
    df.attrs["n_filled"] = n_filled
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode the waste material and scale the numeric inputs."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
