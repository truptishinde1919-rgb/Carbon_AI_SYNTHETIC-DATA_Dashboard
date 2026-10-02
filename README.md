# 🌱 Carbon AI – Waste to Carbon & Electrical Performance

**DEMO DATASET – SYNTHETIC DATA.** All values are synthetic demonstration data created for software
development and project visualization. They are not laboratory measurements and must not be read as
experimentally validated results. Future versions can replace the demo dataset with real lab measurements.

An AI-assisted experimental-data analysis and prediction platform for waste-derived carbon materials and
their potential electrical energy-storage applications (prototype / simulation / demo dashboard).

## Run

```bash
pip install -r requirements.txt
python src/train_models.py      # optional: the app trains automatically if models/ is empty
streamlit run app.py
```

## Structure

| Path | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (8 pages) |
| `src/preprocessing.py` | CSV loading, validation, missing-value handling, shared constants |
| `src/train_models.py` | Trains Random Forest + Linear Regression per target, selects by 5-fold CV R², tests on 20 % hold-out |
| `src/predictions.py` | Turns form inputs into predicted values |
| `models/*.pkl` | Saved models (yield, capacitance, power, energy) + `auxiliary_models.pkl` (resistance, conductivity, voltage, current) |
| `data/carbon_demo_dashboard_data.csv` | Primary dataset (120 synthetic samples) |

## Formulas used (explain these in your demo)

- Carbon Yield (%) = Carbon Mass / Raw Mass × 100 ; Carbon Mass = Raw Mass × Yield / 100
- Power (W) = Voltage × Current
- Energy (Wh) = ½ · C · V² / 3600, with C in F/g, so it is energy per gram of carbon. This is a simplified
  demonstration calculation, not a full supercapacitor characterization.

## Honest notes on the models

Metrics are computed from the data, never hard-coded. On this synthetic set the yield and capacitance models
show moderate skill, while the power and energy models have negative R² (worse than predicting the mean),
because current and voltage in the demo data are not driven by the process inputs. The dashboard shows this
openly; it is a property of the data, not a bug. and this model created using synthetic data it define that if i gote my real data it looks like this.
