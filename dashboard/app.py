"""
Credit Risk Assessment — Interactive Dashboard
================================================
Enter applicant features, score with the trained model, view SHAP reasons.

Run from project root:
  .\\.venv\\Scripts\\python.exe -m streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make package importable when launched via streamlit
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk.config import ARTIFACTS_PATH, METRICS_PATH, MODELS_DIR  # noqa: E402
try:
    from credit_risk.config import PROJECT_STATUS  # noqa: E402
except ImportError:  # older installs / stale process
    PROJECT_STATUS = {
        "lifecycle": "academic_v2_train_val_test",
        "primary_model_pack": "models/full_data",
    }
from credit_risk.explain import explain_decline  # noqa: E402
from credit_risk.predict import (  # noqa: E402
    assess_applicant,
    load_bundle,
    reset_bundle_cache,
)

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
  }

  .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }

  .hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0e7490 100%);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    color: #f8fafc;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.25);
  }
  .hero h1 {
    margin: 0 0 0.35rem 0;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .hero p { margin: 0; opacity: 0.9; font-size: 0.98rem; }

  .metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    height: 100%;
  }
  .metric-card .label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    font-weight: 600;
  }
  .metric-card .value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 0.25rem;
  }
  .metric-card .sub { font-size: 0.82rem; color: #64748b; margin-top: 0.15rem; }

  .decision-approve {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #6ee7b7;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    color: #065f46;
  }
  .decision-decline {
    background: linear-gradient(135deg, #fef2f2, #fee2e2);
    border: 1px solid #fca5a5;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    color: #991b1b;
  }
  .decision-title {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
  }
  .decision-sub { margin: 0; opacity: 0.9; }

  .reason-chip {
    background: #fff7ed;
    border-left: 4px solid #f97316;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.5rem;
    border-radius: 0 10px 10px 0;
    font-size: 0.92rem;
    color: #9a3412;
  }

  div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  }
  div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  div[data-testid="stSidebar"] .stSelectbox label,
  div[data-testid="stSidebar"] .stSlider label,
  div[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }

  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0e7490, #0369a1);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.55rem 1.2rem;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model…")
def get_bundle():
    if not ARTIFACTS_PATH.exists():
        return None
    reset_bundle_cache()
    return load_bundle(path=ARTIFACTS_PATH, force_reload=True)


@st.cache_data
def get_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def age_to_days_birth(age_years: int) -> int:
    return -int(age_years * 365.25)


def years_employed_to_days(years: float) -> int:
    if years <= 0:
        return 365243  # Home Credit missing sentinel
    return -int(years * 365.25)


PRESETS = {
    "Custom": None,
    "Low risk profile": {
        "AMT_INCOME_TOTAL": 280000.0,
        "AMT_CREDIT": 200000.0,
        "AMT_ANNUITY": 14000.0,
        "AMT_GOODS_PRICE": 200000.0,
        "age": 45,
        "years_employed": 12.0,
        "CNT_CHILDREN": 0,
        "EXT_SOURCE_1": 0.72,
        "EXT_SOURCE_2": 0.78,
        "EXT_SOURCE_3": 0.70,
        "CODE_GENDER": "F",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "FLAG_OWN_CAR": "Y",
        "FLAG_OWN_REALTY": "Y",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "OCCUPATION_TYPE": "Managers",
    },
    "High risk profile": {
        "AMT_INCOME_TOTAL": 45000.0,
        "AMT_CREDIT": 850000.0,
        "AMT_ANNUITY": 48000.0,
        "AMT_GOODS_PRICE": 850000.0,
        "age": 24,
        "years_employed": 0.3,
        "CNT_CHILDREN": 2,
        "EXT_SOURCE_1": 0.12,
        "EXT_SOURCE_2": 0.15,
        "EXT_SOURCE_3": 0.10,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Lower secondary",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "With parents",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "N",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "OCCUPATION_TYPE": "Laborers",
    },
    "Mid risk profile": {
        "AMT_INCOME_TOTAL": 135000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 22000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "age": 34,
        "years_employed": 4.0,
        "CNT_CHILDREN": 1,
        "EXT_SOURCE_1": 0.42,
        "EXT_SOURCE_2": 0.48,
        "EXT_SOURCE_3": 0.40,
        "CODE_GENDER": "M",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_INCOME_TYPE": "Working",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "OCCUPATION_TYPE": "Sales staff",
    },
}

EDUCATION_OPTS = [
    "Higher education",
    "Secondary / secondary special",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
INCOME_OPTS = [
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
    "Student",
    "Businessman",
    "Maternity leave",
]
FAMILY_OPTS = [
    "Married",
    "Single / not married",
    "Civil marriage",
    "Separated",
    "Widow",
    "Unknown",
]
HOUSING_OPTS = [
    "House / apartment",
    "With parents",
    "Municipal apartment",
    "Rented apartment",
    "Office apartment",
    "Co-op apartment",
]
OCCUPATION_OPTS = [
    "Laborers",
    "Sales staff",
    "Core staff",
    "Managers",
    "Drivers",
    "High skill tech staff",
    "Accountants",
    "Medicine staff",
    "Security staff",
    "Cooking staff",
    "Cleaning staff",
    "Private service staff",
    "Low-skill Laborers",
    "Waiters/barmen staff",
    "Secretaries",
    "Realty agents",
    "HR staff",
    "IT staff",
    "",
]


def build_gauge(prob: float, threshold: float) -> go.Figure:
    color = "#dc2626" if prob >= threshold else "#059669"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 42, "family": "JetBrains Mono"}},
            title={"text": "Default probability", "font": {"size": 16, "color": "#64748b"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color},
                "bgcolor": "#f1f5f9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "#d1fae5"},
                    {"range": [threshold * 100, 100], "color": "#fecaca"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_shap_chart(top_features: list[dict]) -> go.Figure:
    if not top_features:
        return go.Figure()
    # reverse for horizontal bar (top at top)
    items = list(reversed(top_features))
    names = [f.get("feature", "?") for f in items]
    vals = [float(f.get("shap_value", f.get("contribution", 0))) for f in items]
    colors = ["#dc2626" if v > 0 else "#059669" for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="SHAP contributions (→ right = higher default risk)",
        height=max(320, 40 * len(items) + 80),
        margin=dict(l=10, r=40, t=50, b=20),
        xaxis_title="SHAP value",
        yaxis_title="",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans"),
        showlegend=False,
    )
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    return fig


def build_metrics_bars(metrics: dict) -> go.Figure:
    models = metrics.get("models") or {}
    names, aucs, pras = [], [], []
    for key in ("logistic_regression", "xgboost"):
        m = models.get(key) or {}
        if not m:
            continue
        names.append("Logistic Regression" if key == "logistic_regression" else "XGBoost")
        aucs.append(float(m.get("roc_auc", 0)))
        pras.append(float(m.get("pr_auc", 0)))
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ROC-AUC", x=names, y=aucs, marker_color="#0e7490", text=[f"{v:.3f}" for v in aucs], textposition="outside"))
    fig.add_trace(go.Bar(name="PR-AUC", x=names, y=pras, marker_color="#7c3aed", text=[f"{v:.3f}" for v in pras], textposition="outside"))
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis=dict(range=[0, 1], title="Score"),
        title="Held-out evaluation metrics",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
bundle = get_bundle()
metrics = get_metrics()

with st.sidebar:
    st.markdown("### 💳 Credit Risk Lab")
    st.caption("Interactive scoring — full Home Credit academic model by default.")

    if bundle is None:
        st.error("Model artifacts not found. Train first.")
        st.code(
            "python scripts/run_academic_lifecycle.py\n"
            "# writes models/full_data/",
            language="bash",
        )
        st.stop()

    st.markdown("---")
    st.markdown("**Artifact pack**")
    st.code(str(MODELS_DIR), language="text")
    st.markdown("**Production model**")
    st.write(bundle.production_model.replace("_", " ").title())
    st.markdown("**Default threshold (τ)**")
    st.write(f"{bundle.threshold:.4f}")
    st.markdown("**Features in model**")
    st.write(str(len(bundle.feature_columns)))
    metrics_preview = get_metrics()
    if metrics_preview:
        st.markdown("**Training rows**")
        st.write(str(metrics_preview.get("n_rows_used", "—")))
        st.markdown("**Lifecycle**")
        st.write(str(metrics_preview.get("lifecycle", PROJECT_STATUS.get("lifecycle", "—"))))

    model_choice = st.selectbox(
        "Score with",
        options=["production", "logistic_regression", "xgboost"],
        format_func=lambda x: {
            "production": f"Production ({bundle.production_model})",
            "logistic_regression": "Logistic Regression",
            "xgboost": "XGBoost",
        }[x],
    )
    thr_override = st.slider(
        "Decision threshold override",
        min_value=0.05,
        max_value=0.95,
        value=float(round(bundle.threshold, 2)),
        step=0.01,
        help="Decline if P(default) ≥ threshold",
    )
    top_k = st.slider("SHAP top features", 3, 15, 8)

    st.markdown("---")
    st.caption("Educational demo · not a bank underwriting system")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Credit Risk Assessment Dashboard</h1>
  <p>Enter applicant values → model returns approve/decline, default probability, and SHAP explanations for declines.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_score, tab_metrics, tab_about = st.tabs(
    ["🧾 Score applicant", "📊 Model metrics", "ℹ️ About"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Score
# ---------------------------------------------------------------------------
with tab_score:
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Applicant inputs")
        preset_name = st.selectbox("Load preset profile", list(PRESETS.keys()))
        preset = PRESETS[preset_name] or {}

        # Use session state keys with preset seeding
        def pget(key, default):
            return preset.get(key, default) if preset else default

        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input(
                "Annual income",
                min_value=0.0,
                max_value=5_000_000.0,
                value=float(pget("AMT_INCOME_TOTAL", 135000)),
                step=1000.0,
                format="%.0f",
            )
            credit = st.number_input(
                "Credit amount",
                min_value=0.0,
                max_value=5_000_000.0,
                value=float(pget("AMT_CREDIT", 450000)),
                step=1000.0,
                format="%.0f",
            )
            annuity = st.number_input(
                "Annuity (payment)",
                min_value=0.0,
                max_value=500_000.0,
                value=float(pget("AMT_ANNUITY", 22000)),
                step=500.0,
                format="%.0f",
            )
            goods = st.number_input(
                "Goods price",
                min_value=0.0,
                max_value=5_000_000.0,
                value=float(pget("AMT_GOODS_PRICE", 450000)),
                step=1000.0,
                format="%.0f",
            )
            children = st.number_input(
                "Number of children",
                min_value=0,
                max_value=15,
                value=int(pget("CNT_CHILDREN", 0)),
                step=1,
            )
        with c2:
            age = st.slider("Age (years)", 18, 70, int(pget("age", 34)))
            years_emp = st.slider(
                "Years employed",
                0.0,
                40.0,
                float(pget("years_employed", 4.0)),
                0.1,
            )
            ext1 = st.slider("External source score 1", 0.0, 1.0, float(pget("EXT_SOURCE_1", 0.45)), 0.01)
            ext2 = st.slider("External source score 2", 0.0, 1.0, float(pget("EXT_SOURCE_2", 0.50)), 0.01)
            ext3 = st.slider("External source score 3", 0.0, 1.0, float(pget("EXT_SOURCE_3", 0.40)), 0.01)

        st.markdown("#### Demographics & contract")
        d1, d2, d3 = st.columns(3)
        with d1:
            gender = st.selectbox(
                "Gender",
                ["M", "F"],
                index=0 if pget("CODE_GENDER", "M") == "M" else 1,
            )
            own_car = st.selectbox(
                "Owns car",
                ["Y", "N"],
                index=0 if pget("FLAG_OWN_CAR", "N") == "Y" else 1,
            )
            own_realty = st.selectbox(
                "Owns realty",
                ["Y", "N"],
                index=0 if pget("FLAG_OWN_REALTY", "Y") == "Y" else 1,
            )
        with d2:
            edu_default = pget("NAME_EDUCATION_TYPE", "Secondary / secondary special")
            education = st.selectbox(
                "Education",
                EDUCATION_OPTS,
                index=EDUCATION_OPTS.index(edu_default) if edu_default in EDUCATION_OPTS else 1,
            )
            income_type = st.selectbox(
                "Income type",
                INCOME_OPTS,
                index=INCOME_OPTS.index(pget("NAME_INCOME_TYPE", "Working"))
                if pget("NAME_INCOME_TYPE", "Working") in INCOME_OPTS
                else 0,
            )
            family = st.selectbox(
                "Family status",
                FAMILY_OPTS,
                index=FAMILY_OPTS.index(pget("NAME_FAMILY_STATUS", "Married"))
                if pget("NAME_FAMILY_STATUS", "Married") in FAMILY_OPTS
                else 0,
            )
        with d3:
            housing = st.selectbox(
                "Housing",
                HOUSING_OPTS,
                index=HOUSING_OPTS.index(pget("NAME_HOUSING_TYPE", "House / apartment"))
                if pget("NAME_HOUSING_TYPE", "House / apartment") in HOUSING_OPTS
                else 0,
            )
            contract = st.selectbox(
                "Contract type",
                ["Cash loans", "Revolving loans"],
                index=0 if pget("NAME_CONTRACT_TYPE", "Cash loans") == "Cash loans" else 1,
            )
            occ_default = pget("OCCUPATION_TYPE", "Sales staff")
            occupation = st.selectbox(
                "Occupation",
                OCCUPATION_OPTS,
                index=OCCUPATION_OPTS.index(occ_default) if occ_default in OCCUPATION_OPTS else 0,
            )

        with st.expander("Advanced: JSON feature override (optional)"):
            st.caption("Paste extra raw Home Credit columns as JSON. Merged on top of the form.")
            extra_json = st.text_area("Extra features JSON", value="{}", height=120)

        run = st.button("▶ Run assessment", type="primary", use_container_width=True)

    with right:
        st.subheader("Result")
        if not run:
            st.info("Fill in the form (or pick a preset) and click **Run assessment**.")
        else:
            features = {
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods,
                "CNT_CHILDREN": children,
                "DAYS_BIRTH": age_to_days_birth(age),
                "DAYS_EMPLOYED": years_employed_to_days(years_emp),
                "EXT_SOURCE_1": ext1,
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "NAME_EDUCATION_TYPE": education,
                "NAME_INCOME_TYPE": income_type,
                "NAME_FAMILY_STATUS": family,
                "NAME_HOUSING_TYPE": housing,
                "NAME_CONTRACT_TYPE": contract,
            }
            if occupation:
                features["OCCUPATION_TYPE"] = occupation

            try:
                extra = json.loads(extra_json or "{}")
                if isinstance(extra, dict):
                    features.update(extra)
            except json.JSONDecodeError:
                st.warning("Extra JSON was invalid — ignored.")

            model_name = None if model_choice == "production" else model_choice
            with st.spinner("Scoring applicant…"):
                assessment = assess_applicant(
                    features,
                    bundle=bundle,
                    model_name=model_name,
                    threshold=thr_override,
                )
                full = explain_decline(
                    features,
                    assessment,
                    bundle=bundle,
                    top_k=top_k,
                )

            decision = full["decision"]
            prob = float(full["default_probability"])
            thr = float(full["threshold"])

            if decision == "approve":
                st.markdown(
                    f"""
                    <div class="decision-approve">
                      <p class="decision-title">✅ APPROVE</p>
                      <p class="decision-sub">{full.get("message", "")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="decision-decline">
                      <p class="decision-title">⛔ DECLINE</p>
                      <p class="decision-sub">{full.get("message", "")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="label">P(default)</div>
                      <div class="value">{prob:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="label">Threshold τ</div>
                      <div class="value">{thr:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="label">Model</div>
                      <div class="value" style="font-size:1.05rem">{full.get("model","").replace("_"," ")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.plotly_chart(build_gauge(prob, thr), use_container_width=True)

            expl = full.get("explanation") or {}
            top = expl.get("top_features") or []
            if top:
                st.plotly_chart(build_shap_chart(top), use_container_width=True)

            reasons = full.get("decline_reasons") or expl.get("decline_reasons") or []
            if decision == "decline" and reasons:
                st.markdown("#### Why declined (top risk factors)")
                for r in reasons[:top_k]:
                    st.markdown(f'<div class="reason-chip">{r}</div>', unsafe_allow_html=True)
            elif decision == "approve" and top:
                st.markdown("#### Key drivers of the score")
                st.caption("Even for approvals, SHAP shows which features pushed risk up or down.")
                for item in top[:5]:
                    direction = item.get("direction", "")
                    icon = "🔺" if direction == "increased_risk" else "🔻"
                    st.write(f"{icon} **{item.get('feature')}** — SHAP {item.get('shap_value', 0):+.4f}")

            with st.expander("Raw JSON response"):
                st.json(full)

            # Download payload
            st.download_button(
                "Download result JSON",
                data=json.dumps(full, indent=2, default=str),
                file_name="credit_assessment_result.json",
                mime="application/json",
            )


# ---------------------------------------------------------------------------
# Tab 2 — Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Training evaluation snapshot")
    if not metrics:
        st.warning("No metrics.json found. Train the model first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Production model", str(metrics.get("production_model", "—")).replace("_", " "))
        c2.metric("Threshold", f"{float(metrics.get('threshold', 0)):.3f}")
        c3.metric("Sample size", f"{metrics.get('n_rows_used', metrics.get('sample_size', '—')):,}")
        c4.metric("Features", str(metrics.get("n_features", "—")))

        st.plotly_chart(build_metrics_bars(metrics), use_container_width=True)

        rows = []
        for key, m in (metrics.get("models") or {}).items():
            rows.append(
                {
                    "Model": key,
                    "ROC-AUC": round(float(m.get("roc_auc", 0)), 4),
                    "PR-AUC": round(float(m.get("pr_auc", 0)), 4),
                    "Precision": round(float(m.get("precision", 0)), 4),
                    "Recall": round(float(m.get("recall", 0)), 4),
                    "Train rows": m.get("n_train"),
                    "Test rows": m.get("n_test"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        thr_info = metrics.get("threshold_info") or {}
        if thr_info:
            st.markdown("#### Threshold selection")
            st.json(thr_info)

        st.caption(metrics.get("note", ""))


# ---------------------------------------------------------------------------
# Tab 3 — About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("About this system")
    st.markdown(
        """
**Credit Risk Assessment System** estimates the probability that an applicant will have
payment difficulties (default risk) using Home Credit–style application features.

| Piece | Implementation |
|-------|----------------|
| Models | Logistic Regression + XGBoost |
| Preprocessing | Median/mode impute, scaling (LR), one-hot encoding |
| Decision | Decline if P(default) >= threshold |
| Explainability | SHAP local feature contributions |
| API | FastAPI (`credit_risk.api`) |
| This UI | Streamlit dashboard |

### How to use for a demo
1. Pick a **preset** (low / mid / high risk) or enter your own values.
2. Click **Run assessment**.
3. Show the gauge, decision banner, and SHAP chart.
4. Open **Model metrics** for ROC-AUC / PR-AUC.

### Commands
```powershell
# Dashboard
.\\venv\\Scripts\\python.exe -m streamlit run dashboard/app.py

# API (optional parallel)
.\\venv\\Scripts\\python.exe -m uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000

# Tests
.\\venv\\Scripts\\python.exe -m pytest -q
```

### Academic note
This is an **educational** system. It is not a licensed bank underwriting product.
Metrics in the Metrics tab reflect the last training run (check sample size).
        """
    )
