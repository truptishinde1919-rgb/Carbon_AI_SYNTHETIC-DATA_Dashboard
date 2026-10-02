"""
🌱 Carbon AI – Waste to Carbon & Electrical Performance

Streamlit dashboard (DEMO / SYNTHETIC DATA).   Run with:  streamlit run app.py

Sections of this file
  1. Setup & styling          5. Page functions (one per sidebar item)
  2. Data / model loading     6. Main router
  3. Small UI helpers
  4. Shared input form
"""
import io
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.predictions import out_of_range_warnings, predict_all
from src.preprocessing import DATA_PATH, LABELS, DataError, load_data
from src.train_models import load_models, train_all

# 1. SETUP & STYLING

st.set_page_config(page_title="Carbon AI", page_icon="🌱", layout="wide")

GREEN, DARK_GREEN, BLACK = "#1b9e5a", "#0f6b3c", "#0b1410"
MATERIAL_COLORS = {"Coconut Shell": "#0f6b3c", "Sugarcane Bagasse": "#7bc67b",
                   "Sawdust": "#c9a227", "Corncob": "#4a4a4a"}
MATERIALS = list(MATERIAL_COLORS)

st.markdown(f"""
<style>
  [data-testid="stSidebar"] {{ background:{BLACK}; }}
  [data-testid="stSidebar"] * {{ color:#e8f5ec !important; }}
  .hero {{ background:linear-gradient(120deg,{BLACK},{DARK_GREEN}); color:#fff;
           padding:1.4rem 1.6rem; border-radius:14px; margin-bottom:1rem; }}
  .hero h1 {{ color:#fff; margin:0; font-size:1.9rem; }}
  .hero p  {{ color:#cfe9d8; margin:.3rem 0 .6rem 0; }}
  .badge {{ display:inline-block; background:#ffd84d; color:#000; font-weight:700;
            padding:.2rem .7rem; border-radius:999px; font-size:.8rem; }}
  .kpi {{ background:#f1f7f3; border-left:5px solid {GREEN}; border-radius:10px;
          padding:.8rem 1rem; margin-bottom:.6rem; }}
  .kpi .l {{ font-size:.78rem; color:#4b5d54; text-transform:uppercase; letter-spacing:.04em; }}
  .kpi .v {{ font-size:1.55rem; font-weight:700; color:{BLACK}; }}
  .kpi .u {{ font-size:.85rem; color:#4b5d54; font-weight:400; }}
  .kpi.pred {{ background:{BLACK}; border-left-color:#7bf0a8; }}
  .kpi.pred .l, .kpi.pred .u {{ color:#9fd8b4; }}  .kpi.pred .v {{ color:#fff; }}
  .flow {{ display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; margin:.5rem 0 1rem; }}
  .step {{ background:{BLACK}; color:#fff; padding:.5rem .8rem; border-radius:8px; font-weight:600; font-size:.9rem; }}
  .step.ai {{ background:{GREEN}; }}
  .arrow {{ color:{GREEN}; font-weight:700; }}
  .formula {{ background:#f6fbf8; border:1px dashed {GREEN}; border-radius:10px; padding:.7rem 1rem; }}
</style>
""", unsafe_allow_html=True)

DISCLAIMER = ("All values in the current dashboard are synthetic demonstration data created for "
              "software development and project visualization. They are not laboratory measurements "
              "and should not be interpreted as experimentally validated results.")
FUTURE_NOTE = "Future versions can replace the demo dataset with real laboratory measurements."

# 2. DATA / MODEL LOADING (cached so the app stays fast)

@st.cache_data
def get_data():
    return load_data()


@st.cache_resource(show_spinner="Loading AI/ML models…")
def get_models():
    """Load saved models, or train them from the CSV if the .pkl files are missing."""
    try:
        return load_models(), None
    except Exception:
        pass  # files missing / incompatible -> retrain below
    try:
        return train_all(get_data()), None
    except Exception as exc:
        return None, f"Model training failed: {exc}"


# 3. SMALL UI HELPERS

def show_fig(fig):
    """Display a Plotly chart at full width (works on old and new Streamlit)."""
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), template="plotly_white", height=380)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def kpi(col, label, value, unit="", pred=False):
    col.markdown(f'<div class="kpi {"pred" if pred else ""}"><div class="l">{label}</div>'
                 f'<div class="v">{value} <span class="u">{unit}</span></div></div>',
                 unsafe_allow_html=True)


def flow_diagram(highlight_ai=True):
    steps = ["Waste Material", "Drying", "Carbonization", "Carbon-rich Material", "Activation",
             "Activated Carbon", "Electrode", "Electrical Testing", "AI/ML Prediction"]
    html = '<div class="flow">'
    for i, s in enumerate(steps):
        cls = "step ai" if (highlight_ai and s == "AI/ML Prediction") else "step"
        html += f'<span class="{cls}">{s}</span>' + ('<span class="arrow">➜</span>' if i < len(steps) - 1 else "")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def scatter(df, x, y, title):
    fig = px.scatter(df, x=x, y=y, color="Waste_Material", color_discrete_map=MATERIAL_COLORS,
                     hover_data=["Sample_ID"], labels=LABELS, title=title)
    show_fig(fig)


def bar_avg(df, y, title):
    g = df.groupby("Waste_Material", as_index=False)[y].mean()
    fig = px.bar(g, x="Waste_Material", y=y, color="Waste_Material",
                 color_discrete_map=MATERIAL_COLORS, labels=LABELS, title=title)
    fig.update_layout(showlegend=False)
    show_fig(fig)


def render_filters(df, prefix):
    """Dropdown / slider filters. Returns the filtered DataFrame."""
    c1, c2, c3, c4 = st.columns(4)
    mats = c1.multiselect("Waste Material", sorted(df.Waste_Material.unique()),
                          default=sorted(df.Waste_Material.unique()), key=f"{prefix}_m")
    stat = c2.multiselect("Test Status", sorted(df.Test_Status.unique()),
                          default=sorted(df.Test_Status.unique()), key=f"{prefix}_s")
    t_lo, t_hi = int(df.Carbonization_Temperature_C.min()), int(df.Carbonization_Temperature_C.max())
    temp = c3.slider("Carbonization Temp (°C)", t_lo, t_hi, (t_lo, t_hi), key=f"{prefix}_t")
    m_lo, m_hi = int(df.Carbonization_Time_min.min()), int(df.Carbonization_Time_min.max())
    time_ = c4.slider("Carbonization Time (min)", m_lo, m_hi, (m_lo, m_hi), key=f"{prefix}_c")
    out = df[df.Waste_Material.isin(mats) & df.Test_Status.isin(stat)
             & df.Carbonization_Temperature_C.between(*temp)
             & df.Carbonization_Time_min.between(*time_)]
    st.caption(f"Showing **{len(out)}** of {len(df)} samples")
    return out


def page_header(title):
    st.markdown(f"## {title}")


# 4. SHARED INPUT FORM + RESULT CARDS (used by two pages)

def input_form(prefix):
    """Process-parameter inputs. Returns a dict matching the model's feature names."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Waste material & carbonization**")
        material = st.selectbox("Waste Material", MATERIALS, key=f"{prefix}_mat")
        raw = st.number_input("Raw Material Mass (g)", 100.0, 10000.0, 2000.0, 50.0, key=f"{prefix}_raw")
        moist = st.number_input("Initial Moisture (%)", 0.0, 40.0, 8.0, 0.5, key=f"{prefix}_moist")
        ct = st.slider("Carbonization Temperature (°C)", 300, 1000, 700, 10, key=f"{prefix}_ct")
        ctime = st.slider("Carbonization Time (min)", 5, 300, 90, 5, key=f"{prefix}_ctime")
    with c2:
        st.markdown("**Activation**")
        ratio = st.slider("Activation Ratio (KOH : carbon)", 0.0, 5.0, 1.5, 0.05, key=f"{prefix}_ratio")
        at = st.slider("Activation Temperature (°C)", 400, 1000, 750, 10, key=f"{prefix}_at")
        atime = st.slider("Activation Time (min)", 5, 240, 60, 5, key=f"{prefix}_atime")
    return {"Waste_Material": material, "Raw_Material_Mass_g": raw, "Initial_Moisture_pct": moist,
            "Carbonization_Temperature_C": ct, "Carbonization_Time_min": ctime,
            "Activation_Ratio_KOH_or_Activator": ratio, "Activation_Temperature_C": at,
            "Activation_Time_min": atime}


def run_prediction(models, inputs, state_key):
    """Validate inputs, predict, and store the result in session_state."""
    main, aux = models
    try:
        st.session_state[state_key] = {
            "res": predict_all(main, aux, inputs),
            "warn": out_of_range_warnings(inputs, main["carbon_yield"]),
            "inputs": inputs}
    except Exception as exc:
        st.session_state.pop(state_key, None)
        st.error(f"Prediction failed: {exc}")


def show_results(state_key, full):
    """Display stored prediction. full=True shows every electrical value."""
    store = st.session_state.get(state_key)
    if not store:
        return
    r = store["res"]
    st.markdown("### 🤖 AI/ML Demo Prediction")
    st.caption("Model-generated from synthetic data – not a laboratory measurement.")
    for w in store["warn"]:
        st.warning(w)
    if full:
        rows = [
            [("Carbon Yield", f"{r['yield_pct']:.2f}", "%"), ("Estimated Carbon Mass", f"{r['carbon_mass']:.1f}", "g"),
             ("Capacitance", f"{r['capacitance']:.1f}", "F/g")],
            [("Resistance", f"{r['resistance']:.3f}", "Ω"), ("Conductivity", f"{r['conductivity']:.3f}", "S/cm"),
             ("Voltage", f"{r['voltage']:.2f}", "V")],
            [("Current", f"{r['current']:.3f}", "A"), ("Output Power (V × I)", f"{r['power_formula']:.3f}", "W"),
             ("Energy Storage (½CV²)", f"{r['energy_formula']:.4f}", "Wh")],
        ]
        for row in rows:
            cols = st.columns(3)
            for c, (l, v, u) in zip(cols, row):
                kpi(c, l, v, u, pred=True)
        st.caption(f"Model-based estimates for comparison — Output Power: {r['power_model']:.3f} W, "
                   f"Energy Storage: {r['energy_model']:.4f} Wh "
                   "(see Model Performance: these two models are weak on the demo data).")
    else:
        cols = st.columns(3)
        kpi(cols[0], "Predicted Carbon Yield", f"{r['yield_pct']:.2f}", "%", True)
        kpi(cols[1], "Estimated Carbon Mass", f"{r['carbon_mass']:.1f}", "g", True)
        kpi(cols[2], "Predicted Capacitance", f"{r['capacitance']:.1f}", "F/g", True)
        cols = st.columns(3)
        kpi(cols[0], "Predicted Resistance", f"{r['resistance']:.3f}", "Ω", True)
        kpi(cols[1], "Predicted Output Power", f"{r['power_model']:.3f}", "W", True)
        kpi(cols[2], "Predicted Energy Storage", f"{r['energy_model']:.4f}", "Wh", True)
        st.caption("Carbon Mass = Raw Mass × Predicted Yield ÷ 100. "
                   "Power and energy come from the dedicated Random-Forest/Linear models.")


# 5. PAGES

def page_overview(df, models):
    page_header("🏠 Overview")
    cols = st.columns(4)
    kpi(cols[0], "Total Samples", len(df))
    kpi(cols[1], "Avg Carbon Yield", f"{df.Carbon_Yield_pct.mean():.2f}", "%")
    kpi(cols[2], "Avg Capacitance", f"{df.Capacitance_F_per_g.mean():.1f}", "F/g")
    kpi(cols[3], "Avg Resistance", f"{df.Resistance_Ohm.mean():.3f}", "Ω")
    cols = st.columns(3)
    kpi(cols[0], "Avg Conductivity", f"{df.Conductivity_S_per_cm.mean():.3f}", "S/cm")
    kpi(cols[1], "Avg Output Power", f"{df.Output_Power_W.mean():.3f}", "W")
    kpi(cols[2], "Avg Energy Storage", f"{df.Energy_Storage_Wh.mean():.4f}", "Wh")

    st.markdown("### Project Process")
    flow_diagram()
    st.info("The system analyzes demo carbon-production and electrical-performance data to identify "
            "relationships between processing conditions, carbon yield and electrical properties.")

    st.markdown("### About the Project")
    st.write("The project explores the conversion of waste materials into carbon-rich materials through "
             "controlled carbonization and activation. The produced carbon can then be processed into an "
             "electrode and evaluated through electrical parameters such as voltage, current, resistance, "
             "conductivity and capacitance. AI/ML is used to analyze relationships between processing "
             "conditions and electrical performance.")
    st.write("**Positioning:** an AI-assisted experimental-data analysis and prediction platform for "
             "waste-derived carbon materials and their potential electrical energy-storage applications. "
             "The current version is a **prototype / simulation / demo dashboard**, not a validated industrial system.")
    st.warning(DISCLAIMER + " " + FUTURE_NOTE)


def page_process(df, models):
    page_header("🧪 Process Simulation")
    flow_diagram()
    inputs = input_form("sim")
    if st.button("▶ Run Carbon Process", type="primary"):
        run_prediction(models, inputs, "sim_result")
    show_results("sim_result", full=True)

    with st.expander("📐 Carbon yield calculation (for explaining the demo)", expanded=True):
        st.markdown('<div class="formula">Carbon Yield (%) = (Carbon Mass ÷ Raw Material Mass) × 100<br>'
                    'Carbon Mass = Raw Material Mass × Carbon Yield ÷ 100</div>', unsafe_allow_html=True)
        store = st.session_state.get("sim_result")
        if store:
            raw, y, m = store["inputs"]["Raw_Material_Mass_g"], store["res"]["yield_pct"], store["res"]["carbon_mass"]
            st.write(f"Your run: {raw:,.0f} g × {y:.2f} ÷ 100 = **{m:,.1f} g** carbon")
        else:
            s = df.iloc[0]
            st.write(f"Dataset example ({s.Sample_ID}): {s.Raw_Material_Mass_g:,.1f} g × {s.Carbon_Yield_pct} ÷ 100 "
                     f"≈ {s.Raw_Material_Mass_g * s.Carbon_Yield_pct / 100:,.1f} g (CSV value: {s.Carbon_Mass_g} g)")


def page_analysis(df, models):
    page_header("📊 Data Analysis")
    f = render_filters(df, "an")
    if f.empty:
        st.warning("No samples match the filters. Widen the filter ranges.")
        return
    c1, c2 = st.columns(2)
    with c1:
        scatter(f, "Carbonization_Temperature_C", "Carbon_Yield_pct", "1 · Carbonization Temperature vs Carbon Yield")
        bar_avg(f, "Carbon_Yield_pct", "3 · Waste Material vs Average Carbon Yield")
        scatter(f, "Capacitance_F_per_g", "Resistance_Ohm", "5 · Capacitance vs Resistance")
        scatter(f, "Current_A", "Output_Power_W", "7 · Current vs Output Power")
    with c2:
        scatter(f, "Carbonization_Time_min", "Carbon_Yield_pct", "2 · Carbonization Time vs Carbon Yield")
        scatter(f, "Activation_Temperature_C", "Capacitance_F_per_g", "4 · Activation Temperature vs Capacitance")
        scatter(f, "Voltage_V", "Current_A", "6 · Voltage vs Current")
        scatter(f, "Capacitance_F_per_g", "Energy_Storage_Wh", "8 · Capacitance vs Energy Storage")


def page_electrical(df, models):
    page_header("⚡ Electrical Performance")
    f = render_filters(df, "el")
    if f.empty:
        st.warning("No samples match the filters.")
        return
    cols = st.columns(4)
    kpi(cols[0], "Voltage", f"{f.Voltage_V.mean():.2f}", "V")
    kpi(cols[1], "Current", f"{f.Current_A.mean():.3f}", "A")
    kpi(cols[2], "Resistance", f"{f.Resistance_Ohm.mean():.3f}", "Ω")
    kpi(cols[3], "Conductivity", f"{f.Conductivity_S_per_cm.mean():.3f}", "S/cm")
    cols = st.columns(3)
    kpi(cols[0], "Capacitance", f"{f.Capacitance_F_per_g.mean():.1f}", "F/g")
    kpi(cols[1], "Output Power", f"{f.Output_Power_W.mean():.3f}", "W")
    kpi(cols[2], "Energy Storage", f"{f.Energy_Storage_Wh.mean():.4f}", "Wh")
    st.caption("KPI cards show averages of the filtered samples.")

    t1, t2, t3 = st.tabs(["Voltage · Current · Power", "Material comparison", "Formulas"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            scatter(f, "Voltage_V", "Current_A", "Voltage–Current Relationship")
        with c2:
            scatter(f, "Current_A", "Output_Power_W", "Power Analysis (Current vs Output Power)")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(f, x="Waste_Material", y="Energy_Storage_Wh", color="Waste_Material", points="all",
                         color_discrete_map=MATERIAL_COLORS, labels=LABELS, title="Energy Storage by Waste Material")
            fig.update_layout(showlegend=False)
            show_fig(fig)
            bar_avg(f, "Resistance_Ohm", "Resistance Comparison (average)")
        with c2:
            bar_avg(f, "Capacitance_F_per_g", "Capacitance Comparison (average)")
    with t3:
        s = df[df.Sample_ID == st.selectbox("Worked example – choose a sample", df.Sample_ID)].iloc[0]
        st.markdown("**Power calculation**")
        st.markdown('<div class="formula">Power (W) = Voltage (V) × Current (A)</div>', unsafe_allow_html=True)
        st.write(f"{s.Voltage_V} V × {s.Current_A} A = **{s.Voltage_V * s.Current_A:.3f} W** "
                 f"(CSV value: {s.Output_Power_W} W)")
        st.markdown("**Energy-storage calculation (simplified demonstration)**")
        st.markdown('<div class="formula">Energy (Wh) = ½ × C × V² ÷ 3600 &nbsp;(C in F/g, V in volts)</div>',
                    unsafe_allow_html=True)
        st.write(f"0.5 × {s.Capacitance_F_per_g} × {s.Voltage_V}² ÷ 3600 = "
                 f"**{0.5 * s.Capacitance_F_per_g * s.Voltage_V ** 2 / 3600:.4f} Wh** (CSV value: {s.Energy_Storage_Wh} Wh)")
        st.info("This is a simplified demonstration calculation (energy per gram of carbon). "
                "It is not a complete laboratory supercapacitor characterization.")


def page_prediction(df, models):
    page_header("🤖 AI Prediction")
    st.caption("Random Forest and Linear Regression were trained on the synthetic demo dataset.")
    inputs = input_form("pred")
    if st.button("🔮 Predict Carbon Performance", type="primary"):
        run_prediction(models, inputs, "pred_result")
    show_results("pred_result", full=False)
    st.info("Predictions are model estimates learned from synthetic data. They are NOT experimentally validated.")


def page_model_perf(df, models):
    page_header("📈 Model Performance")
    main, _ = models
    names = {"carbon_yield": "Carbon Yield Model", "capacitance": "Capacitance Model",
             "power": "Power Model", "energy": "Energy Model"}
    st.caption("Metrics are computed on a 20 % hold-out test split of the demo dataset. The better of the two "
               "algorithms is chosen by 5-fold cross-validation R² on the training split. "
               "R² < 0 means the model is worse than simply predicting the average.")
    summary = pd.DataFrame([{"Model": names[k], "Selected algorithm": b["model_name"], "R²": b["metrics"]["R2"],
                             "MAE": b["metrics"]["MAE"], "RMSE": b["metrics"]["RMSE"]} for k, b in main.items()])
    show_table(summary.round(4), hide_index=True)

    tabs = st.tabs(list(names.values()))
    for tab, (k, title) in zip(tabs, names.items()):
        b = main[k]
        with tab:
            m = b["metrics"]
            cols = st.columns(3)
            kpi(cols[0], "R²", f"{m['R2']:.3f}")
            kpi(cols[1], "MAE", f"{m['MAE']:.4f}")
            kpi(cols[2], "RMSE", f"{m['RMSE']:.4f}")
            st.caption(f"Selected: **{b['model_name']}** · trained on {b['n_train']} samples, tested on {b['n_test']}.")
            if m["R2"] < 0.3:
                st.warning("Weak predictive power on this dataset – treat predictions for this target as illustrative only.")
            cmp = pd.DataFrame(b["all_metrics"]).T.rename(columns={"CV_R2": "CV R² (selection)", "R2": "Test R²"})
            show_table(cmp.round(4))
            c1, c2 = st.columns(2)
            with c1:
                lo, hi = min(b["y_test"].min(), b["y_pred"].min()), max(b["y_test"].max(), b["y_pred"].max())
                fig = px.scatter(x=b["y_test"], y=b["y_pred"], labels={"x": "Actual", "y": "Predicted"},
                                 title="Actual vs Predicted (test set)", color_discrete_sequence=[GREEN])
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect fit",
                                         line=dict(color=BLACK, dash="dash")))
                show_fig(fig)
            with c2:
                imp = b["importance"].copy()
                imp["Feature"] = imp["Feature"].map(lambda f: LABELS.get(f, f))
                fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                             title="Random Forest Feature Importance", color_discrete_sequence=[GREEN])
                show_fig(fig)
    st.warning("These metrics describe performance on synthetic data only; the models are not experimentally validated.")


def page_explorer(df, models):
    page_header("🔍 Sample Explorer")
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 1])
    q = c1.text_input("Search Sample_ID", placeholder="e.g. CARB-001")
    mats = c2.multiselect("Waste Material", sorted(df.Waste_Material.unique()))
    stat = c3.multiselect("Test Status", sorted(df.Test_Status.unique()))
    sort_col = c4.selectbox("Sort by", ["Sample_ID", "Carbon_Yield_pct", "Capacitance_F_per_g",
                                         "Resistance_Ohm", "Output_Power_W"], format_func=lambda c: LABELS.get(c, c))
    asc = c5.checkbox("Ascending", value=True)
    f = df.copy()
    if q:
        f = f[f.Sample_ID.str.contains(q, case=False, na=False)]
    if mats:
        f = f[f.Waste_Material.isin(mats)]
    if stat:
        f = f[f.Test_Status.isin(stat)]
    f = f.sort_values(sort_col, ascending=asc)
    st.caption(f"{len(f)} samples")
    show_table(f, hide_index=True, height=300)
    if f.empty:
        st.info("No matching samples.")
        return
    sid = st.selectbox("Select a sample to inspect", f.Sample_ID)
    s = df[df.Sample_ID == sid].iloc[0]
    st.markdown(f"### Sample {sid}  ·  {s.Waste_Material}  ·  Status: {s.Test_Status}")
    groups = {"Process": ["Raw_Material_Mass_g", "Initial_Moisture_pct", "Carbonization_Temperature_C",
                          "Carbonization_Time_min", "Activation_Ratio_KOH_or_Activator",
                          "Activation_Temperature_C", "Activation_Time_min"],
              "Carbon result": ["Carbon_Yield_pct", "Carbon_Mass_g"],
              "Electrical": ["Capacitance_F_per_g", "Resistance_Ohm", "Conductivity_S_per_cm", "Voltage_V",
                             "Current_A", "Output_Power_W", "Energy_Storage_Wh"]}
    for name, cols_ in groups.items():
        st.markdown(f"**{name}**")
        cols = st.columns(4)
        for i, c in enumerate(cols_):
            kpi(cols[i % 4], LABELS[c], f"{s[c]:g}")


def page_download(df, models):
    page_header("📥 Download Data")
    st.markdown("**Filter the records you want to download:**")
    f = render_filters(df, "dl")
    c1, c2, c3 = st.columns(3)
    try:
        original = Path(DATA_PATH).read_bytes()
    except OSError as exc:
        st.error(f"Could not read the original CSV: {exc}")
        return
    c1.download_button("⬇ Download Complete Dataset (CSV)", original, "carbon_demo_dashboard_data.csv", "text/csv")
    c2.download_button("⬇ Download Filtered Dataset (CSV)", f.to_csv(index=False).encode(),
                       "carbon_filtered_data.csv", "text/csv", disabled=f.empty)
    try:
        buf = io.BytesIO()
        f.to_excel(buf, index=False, sheet_name="Filtered")
        c3.download_button("⬇ Filtered Dataset (Excel)", buf.getvalue(), "carbon_filtered_data.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", disabled=f.empty)
    except Exception as exc:  # openpyxl missing etc.
        c3.warning(f"Excel export unavailable: {exc}")
    show_table(f, hide_index=True, height=300)

# 6. MAIN ROUTER

PAGES = {"🏠 Overview": page_overview, "🧪 Process Simulation": page_process,
         "📊 Data Analysis": page_analysis, "⚡ Electrical Performance": page_electrical,
         "🤖 AI Prediction": page_prediction, "📈 Model Performance": page_model_perf,
         "🔍 Sample Explorer": page_explorer, "📥 Download Data": page_download}


def main():
    with st.sidebar:
        st.markdown("## 🌱 Carbon AI")
        choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.markdown("---")
        st.caption("Prototype / Simulation / Demo Dashboard")

    st.markdown('<div class="hero"><h1>🌱 Carbon AI – Waste to Carbon &amp; Electrical Performance</h1>'
                '<p>AI-powered analysis of biomass-derived carbon for energy-storage applications</p>'
                '<span class="badge">DEMO DATASET – SYNTHETIC DATA</span></div>', unsafe_allow_html=True)

    try:
        df = get_data()
    except DataError as exc:
        st.error(f"❌ {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"❌ Unexpected error while loading data: {exc}")
        st.stop()
    if df.attrs.get("n_filled"):
        st.info(f"{df.attrs['n_filled']} missing/invalid numeric values were replaced with column medians.")

    models, err = get_models()
    if err:  # never fake metrics – show the error and stop model pages
        st.error(f"❌ {err}")
        if choice in ("🧪 Process Simulation", "🤖 AI Prediction", "📈 Model Performance"):
            st.stop()

    PAGES[choice](df, models)
    st.markdown("---")
    st.caption(DISCLAIMER + " " + FUTURE_NOTE)


main()
