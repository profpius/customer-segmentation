"""
Customer Segmentation App
RFM-based KMeans clustering — production-ready Streamlit dashboard
"""

import io
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segments",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1f1f1f;
}
section[data-testid="stSidebar"] * {
    color: #e5e5e5 !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.93rem;
    padding: 6px 0;
    cursor: pointer;
}
section[data-testid="stSidebar"] hr {
    border-color: #2a2a2a;
}

/* ── Main background ── */
.main .block-container {
    background: #f7f6f2;
    padding: 2rem 2.5rem 3rem;
}

/* ── Display heading ── */
.display-heading {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #0d0d0d;
    margin-bottom: 0.25rem;
}
.display-sub {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    font-size: 1.05rem;
    color: #666;
    margin-bottom: 2rem;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.6rem;
}

/* ── Cards ── */
.stat-card {
    background: #fff;
    border: 1px solid #e8e8e4;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.stat-card h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #0d0d0d;
    margin: 0 0 2px;
}
.stat-card p {
    font-size: 0.82rem;
    color: #888;
    margin: 0;
}

/* ── Segment badge ── */
.segment-badge {
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.4rem 1.1rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}

/* ── Insight block ── */
.insight-block {
    background: #fff;
    border: 1px solid #e8e8e4;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid #0d0d0d;
}
.insight-block h4 {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 0 0 0.5rem;
    color: #0d0d0d;
}
.insight-block p, .insight-block li {
    font-size: 0.88rem;
    color: #444;
    line-height: 1.65;
}

/* ── Divider ── */
.thin-divider {
    border: none;
    border-top: 1px solid #e0e0da;
    margin: 1.8rem 0;
}

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #e8e8e4;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
    color: #0d0d0d !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0d0d0d;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #2a2a2a;
    color: #fff;
}

/* ── Number inputs ── */
.stNumberInput input {
    border-radius: 8px;
    border: 1px solid #d8d8d4;
    font-family: 'DM Sans', sans-serif;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #f0f0ea;
    color: #0d0d0d;
    border: 1px solid #d0d0ca;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SEGMENT CONFIG  (colours, descriptions, recs)
# ─────────────────────────────────────────────
SEGMENT_CONFIG = {
    "Champions": {
        "color": "#1a472a",
        "bg": "#d4edda",
        "emoji": "🏆",
        "description": "Bought recently, buy often, and spend the most. Your most valuable customers.",
        "behavior": (
            "High recency, high frequency, high monetary value. "
            "They transact consistently and respond well to premium offers."
        ),
        "recommendations": [
            "🎁 Reward with exclusive loyalty perks or early product access.",
            "📣 Ask for reviews — they are your best brand ambassadors.",
            "🛍️ Upsell premium lines — high willingness to pay.",
            "🤝 Invite to referral or VIP programmes.",
        ],
    },
    "Promising": {
        "color": "#1a4472",
        "bg": "#d0e4f7",
        "emoji": "🌱",
        "description": "Purchased recently but infrequently. Shows strong potential to become Champions.",
        "behavior": (
            "Moderate recency, low-to-mid frequency, moderate spend. "
            "They are engaged but haven't yet formed a habit."
        ),
        "recommendations": [
            "📬 Send personalised follow-up emails after purchase.",
            "💡 Showcase product categories they haven't explored.",
            "🎟️ Offer a loyalty-tier incentive to encourage repeat buying.",
            "📦 Bundle offers to increase basket size.",
        ],
    },
    "At Risk": {
        "color": "#7d4700",
        "bg": "#fff3cd",
        "emoji": "⚠️",
        "description": "Were good customers but haven't returned in a while. Need re-engagement.",
        "behavior": (
            "Low recency, mid-to-high historical frequency. "
            "They used to buy but have drifted — likely comparing alternatives."
        ),
        "recommendations": [
            "📩 Send a 'We miss you' campaign with a discount.",
            "🔁 Highlight new arrivals since their last order.",
            "📊 Analyse last purchase category for targeted messaging.",
            "⏰ Create urgency with limited-time win-back offers.",
        ],
    },
    "Lost / Inactive": {
        "color": "#6b1c1c",
        "bg": "#f8d7da",
        "emoji": "💤",
        "description": "Haven't purchased in a very long time. Low recency, frequency, and spend.",
        "behavior": (
            "Very low recency and frequency. "
            "Reactivation is expensive — focus only on high-LTV prospects."
        ),
        "recommendations": [
            "💌 Send a final re-engagement email with a strong incentive.",
            "🔍 Survey them to understand why they left.",
            "🗑️ Suppress from regular campaigns to reduce cost.",
            "📉 Mark as churned if no response after re-engagement attempt.",
        ],
    },
}

# Colour palette for cluster plots (up to 4 clusters)
CLUSTER_COLORS = ["#2d6a4f", "#1e3a5f", "#b45309", "#7c1d1d"]


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource(show_spinner=False)
def load_models():
    """Load KMeans model, scaler, and label map from disk."""
    try:
        kmeans    = joblib.load(MODEL_DIR / "kmeans_model.joblib")
        scaler    = joblib.load(MODEL_DIR / "scaler.joblib")
        label_map = joblib.load(MODEL_DIR / "segment_label_map.joblib")
        return kmeans, scaler, label_map, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, str(e)


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict_segment(recency: float, frequency: float, monetary: float,
                    kmeans, scaler, label_map) -> dict:
    """
    Scale input → predict cluster → map to segment name.
    Returns a dict with cluster index, segment name, and colour config.
    """
    input_df = pd.DataFrame(
        [[recency, frequency, monetary]],
        columns=["Recency", "Frequency", "Monetary"]
    )
    # Scaler was fitted on log-transformed columns named Recency_log, Frequency_log, Monetary_log
    input_log = np.log1p(input_df)
    input_log.columns = ["Recency_log", "Frequency_log", "Monetary_log"]
    scaled     = scaler.transform(input_log)
    cluster_id = int(kmeans.predict(scaled)[0])
    segment    = label_map.get(cluster_id, f"Cluster {cluster_id}")
    config     = SEGMENT_CONFIG.get(segment, {
        "color": "#333", "bg": "#eee", "emoji": "📊",
        "description": "Segment identified by the model.",
        "behavior": "—", "recommendations": [],
    })
    return {
        "cluster_id": cluster_id,
        "segment":    segment,
        "config":     config,
        "scaled":     scaled,
        "input_df":   input_df,
    }


# ─────────────────────────────────────────────
# SYNTHETIC CLUSTER DATA (for visualisations)
# ─────────────────────────────────────────────
@st.cache_data
def generate_demo_rfm(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """
    Generate representative synthetic RFM data for each segment
    so charts always render even without the original dataset.
    """
    rng = np.random.default_rng(seed)
    segments = {
        "Champions":      dict(r=(5,30),   f=(8,25),  m=(800,3000), n=n//5),
        "Promising":      dict(r=(10,60),  f=(2,8),   m=(150,700),  n=n//4),
        "At Risk":        dict(r=(90,200), f=(3,12),  m=(200,900),  n=n//4),
        "Lost / Inactive":dict(r=(200,365),f=(1,4),   m=(30,250),   n=n//3),
    }
    rows = []
    for seg, p in segments.items():
        rows.append(pd.DataFrame({
            "Recency":   rng.integers(*p["r"], size=p["n"]),
            "Frequency": rng.integers(*p["f"], size=p["n"]),
            "Monetary":  rng.uniform(*p["m"],  size=p["n"]).round(2),
            "Segment":   seg,
        }))
    return pd.concat(rows, ignore_index=True)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────
def _fig_style():
    """Apply consistent minimal style to matplotlib figures."""
    plt.rcParams.update({
        "figure.facecolor": "#ffffff",
        "axes.facecolor":   "#ffffff",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid":        True,
        "grid.color":       "#eeeeea",
        "grid.linewidth":   0.8,
        "font.family":      "sans-serif",
        "xtick.color":      "#888",
        "ytick.color":      "#888",
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
    })


def chart_scatter(df: pd.DataFrame, highlight: dict | None = None):
    """Frequency vs Monetary scatter coloured by segment."""
    _fig_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    seg_list = list(SEGMENT_CONFIG.keys())

    for i, seg in enumerate(seg_list):
        sub = df[df["Segment"] == seg]
        ax.scatter(sub["Frequency"], sub["Monetary"],
                   color=SEGMENT_CONFIG[seg]["bg"],
                   edgecolors=SEGMENT_CONFIG[seg]["color"],
                   linewidths=0.8, s=28, alpha=0.75, label=seg)

    if highlight:
        ax.scatter(highlight["frequency"], highlight["monetary"],
                   color="#ff4b4b", edgecolors="#0d0d0d",
                   s=160, zorder=9, linewidths=1.5,
                   marker="*", label="Your Customer")

    ax.set_xlabel("Frequency (orders)", fontsize=10, color="#444")
    ax.set_ylabel("Monetary (£)", fontsize=10, color="#444")
    ax.set_title("Frequency vs Monetary by Segment", fontsize=11,
                 fontweight="bold", color="#0d0d0d", pad=12)
    ax.legend(fontsize=8, framealpha=0, loc="upper left")
    fig.tight_layout()
    return fig


def chart_distribution(df: pd.DataFrame):
    """Horizontal bar chart — customer count per segment."""
    _fig_style()
    counts = df["Segment"].value_counts()
    order  = [s for s in ["Lost / Inactive", "At Risk", "Promising", "Champions"]
               if s in counts.index]
    counts = counts.reindex(order)
    colors = [SEGMENT_CONFIG[s]["color"] for s in order]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(counts.index, counts.values, color=colors,
                   height=0.55, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", ha="left", fontsize=9, color="#444")
    ax.set_xlabel("Number of customers", fontsize=9, color="#888")
    ax.set_title("Cluster Distribution", fontsize=11,
                 fontweight="bold", color="#0d0d0d", pad=12)
    ax.set_xlim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig


def chart_pca(df: pd.DataFrame, highlight_row: pd.DataFrame | None = None):
    """PCA 2-D projection of RFM features coloured by segment."""
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.decomposition import PCA

    _fig_style()
    features = ["Recency", "Frequency", "Monetary"]
    X = SS().fit_transform(df[features])
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                            "Segment": df["Segment"].values})
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seg in SEGMENT_CONFIG:
        sub = pca_df[pca_df["Segment"] == seg]
        ax.scatter(sub["PC1"], sub["PC2"],
                   color=SEGMENT_CONFIG[seg]["bg"],
                   edgecolors=SEGMENT_CONFIG[seg]["color"],
                   linewidths=0.7, s=22, alpha=0.7, label=seg)

    if highlight_row is not None:
        from sklearn.preprocessing import StandardScaler as SS2
        sc2 = SS2().fit(df[features])
        h_coords = pca.transform(sc2.transform(highlight_row[features]))
        ax.scatter(h_coords[0, 0], h_coords[0, 1],
                   color="#ff4b4b", edgecolors="#0d0d0d", s=200,
                   marker="*", zorder=10, linewidths=1.5, label="Your Customer")

    ax.set_xlabel(f"PC1 ({var[0]:.1f}% var.)", fontsize=9, color="#888")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}% var.)", fontsize=9, color="#888")
    ax.set_title("PCA — RFM Space", fontsize=11,
                 fontweight="bold", color="#0d0d0d", pad=12)
    ax.legend(fontsize=8, framealpha=0)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
def sidebar_nav() -> str:
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1.2rem 0 0.5rem;'>
            <span style='font-family:Syne,sans-serif;font-weight:800;
                         font-size:1.25rem;color:#fff;letter-spacing:-0.02em;'>
                🧭 SegmentIQ
            </span><br>
            <span style='font-size:0.75rem;color:#666;'>
                RFM · KMeans · Customer Intelligence
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["🏠  Overview", "🔍  Predict Segment", "📂  Batch Predict",
             "📊  Cluster Insights", "📈  Visualisations"],
            label_visibility="collapsed",
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.72rem;color:#555;line-height:1.8;padding-bottom:1rem;'>
            <b style='color:#888;'>Model</b><br>
            KMeans · K=4<br>
            <b style='color:#888;'>Features</b><br>
            Recency · Frequency · Monetary<br>
            <b style='color:#888;'>Segments</b><br>
            Champions · Promising<br>
            At Risk · Lost / Inactive
        </div>
        """, unsafe_allow_html=True)
    return page


# ─────────────────────────────────────────────
# PAGE — OVERVIEW
# ─────────────────────────────────────────────
def page_overview(df: pd.DataFrame):
    st.markdown('<p class="section-label">Customer Intelligence Platform</p>',
                unsafe_allow_html=True)
    st.markdown('<h1 class="display-heading">Segment your<br>customers.</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="display-sub">RFM analysis powered by KMeans clustering — '
                'predict which segment any customer belongs to in seconds.</p>',
                unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # KPI row
    st.markdown('<p class="section-label">Dataset snapshot</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers",  f"{len(df):,}")
    c2.metric("Segments",   "4")
    c3.metric("Avg Recency", f"{df['Recency'].mean():.0f} days")
    c4.metric("Avg Monetary", f"£{df['Monetary'].mean():,.0f}")

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # Segment cards
    st.markdown('<p class="section-label">The four segments</p>',
                unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (seg, cfg) in zip(cols, SEGMENT_CONFIG.items()):
        with col:
            cnt = len(df[df["Segment"] == seg])
            pct = cnt / len(df) * 100
            st.markdown(f"""
            <div class="stat-card" style="border-left: 5px solid {cfg['color']};">
                <div style='font-size:1.6rem;margin-bottom:0.3rem;'>{cfg['emoji']}</div>
                <h3 style='font-size:1rem;'>{seg}</h3>
                <p style='font-size:1.4rem;font-family:Syne,sans-serif;
                          font-weight:700;color:{cfg['color']};margin:4px 0;'>
                    {cnt:,} <span style='font-size:0.85rem;color:#999;'>({pct:.0f}%)</span>
                </p>
                <p>{cfg['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">How it works</p>',
                unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)
    for col, step in zip([h1, h2, h3, h4], [
        ("01", "Input RFM", "Enter Recency, Frequency, and Monetary values for a customer."),
        ("02", "Scale",     "The saved StandardScaler normalises the features."),
        ("03", "Predict",   "KMeans assigns the customer to the nearest cluster centroid."),
        ("04", "Insight",   "The cluster maps to a named segment with business actions."),
    ]):
        col.markdown(f"""
        <div style='padding:1rem 0;'>
            <span style='font-family:Syne,sans-serif;font-weight:800;
                         font-size:2rem;color:#d0d0ca;'>{step[0]}</span>
            <h4 style='font-family:Syne,sans-serif;font-weight:700;
                       margin:4px 0 6px;font-size:0.95rem;'>{step[1]}</h4>
            <p style='font-size:0.82rem;color:#666;line-height:1.55;'>{step[2]}</p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE — PREDICT
# ─────────────────────────────────────────────
def page_predict(kmeans, scaler, label_map, df: pd.DataFrame):
    st.markdown('<p class="section-label">Predict</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="display-heading">Segment a customer.</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="display-sub">Enter the customer\'s RFM values below '
                'and the model will instantly classify them.</p>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="section-label">RFM Input</p>', unsafe_allow_html=True)

        # Sample button pre-fills session state
        if st.button("⚡ Load sample customer"):
            st.session_state["recency"]   = 18
            st.session_state["frequency"] = 12
            st.session_state["monetary"]  = 1540.0

        recency   = st.number_input("Recency (days since last purchase)",
                                    min_value=0, max_value=1000, value=st.session_state.get("recency", 60),
                                    step=1, key="recency",
                                    help="Smaller = more recent = better.")
        frequency = st.number_input("Frequency (number of orders)",
                                    min_value=0, max_value=1000, value=st.session_state.get("frequency", 5),
                                    step=1, key="frequency",
                                    help="Higher = more loyal.")
        monetary  = st.number_input("Monetary (total spend £)",
                                    min_value=0.0, max_value=1_000_000.0,
                                    value=float(st.session_state.get("monetary", 350.0)),
                                    step=10.0, key="monetary",
                                    help="Higher = more valuable.")

        predict_btn = st.button("🔍 Predict Segment", use_container_width=True)

    with right:
        if predict_btn or st.session_state.get("last_result"):
            if predict_btn:
                if recency == 0 and frequency == 0 and monetary == 0.0:
                    st.error("⚠️ Please enter at least one non-zero RFM value.")
                    st.session_state.pop("last_result", None)
                    return

                result = predict_segment(recency, frequency, monetary,
                                         kmeans, scaler, label_map)
                st.session_state["last_result"] = result

            result = st.session_state.get("last_result")
            if not result:
                return

            cfg = result["config"]
            seg = result["segment"]

            st.markdown('<p class="section-label">Result</p>',
                        unsafe_allow_html=True)

            # Segment badge
            st.markdown(f"""
            <div style='background:{cfg["bg"]};border-radius:14px;
                        padding:1.4rem 1.8rem;margin-bottom:1rem;
                        border:1px solid {cfg["color"]}33;'>
                <span style='font-size:2.2rem;'>{cfg["emoji"]}</span>
                <h2 style='font-family:Syne,sans-serif;font-weight:800;
                           font-size:1.8rem;color:{cfg["color"]};
                           margin:6px 0 4px;'>{seg}</h2>
                <span style='font-size:0.82rem;color:#555;'>
                    Cluster {result["cluster_id"]}
                </span>
                <p style='margin:10px 0 0;font-size:0.9rem;
                          color:#333;line-height:1.6;'>{cfg["description"]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Input recap
            m1, m2, m3 = st.columns(3)
            m1.metric("Recency",   f"{recency} days")
            m2.metric("Frequency", f"{frequency} orders")
            m3.metric("Monetary",  f"£{monetary:,.0f}")

            st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

            # Download result
            out = pd.DataFrame({
                "Recency":   [recency],
                "Frequency": [frequency],
                "Monetary":  [monetary],
                "Cluster":   [result["cluster_id"]],
                "Segment":   [seg],
            })
            csv_bytes = out.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download prediction as CSV",
                data=csv_bytes,
                file_name="segment_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

            # Scatter showing where this customer sits
            st.markdown('<p class="section-label" style="margin-top:1.2rem;">'
                        'Position in cluster space</p>', unsafe_allow_html=True)
            fig = chart_scatter(df, highlight={"frequency": frequency, "monetary": monetary})
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.markdown("""
            <div style='padding:3rem 1rem;text-align:center;color:#bbb;'>
                <div style='font-size:3rem;'>🧭</div>
                <p style='font-size:0.9rem;margin-top:1rem;'>
                    Fill in the RFM values and click<br>
                    <b>Predict Segment</b> to see results.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE — CLUSTER INSIGHTS
# ─────────────────────────────────────────────
def page_insights(df: pd.DataFrame):
    st.markdown('<p class="section-label">Insights</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="display-heading">Segment playbook.</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="display-sub">Understand each customer segment and '
                'the business actions that drive results.</p>',
                unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # Segment filter
    selected = st.multiselect(
        "Show segments",
        options=list(SEGMENT_CONFIG.keys()),
        default=list(SEGMENT_CONFIG.keys()),
    )

    for seg in selected:
        cfg = SEGMENT_CONFIG[seg]
        sub = df[df["Segment"] == seg]

        r_avg = sub["Recency"].mean()
        f_avg = sub["Frequency"].mean()
        m_avg = sub["Monetary"].mean()
        count = len(sub)
        pct   = count / len(df) * 100

        st.markdown(f"""
        <div class="insight-block" style="border-left-color:{cfg['color']};">
            <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;'>
                <span style='font-size:1.8rem;'>{cfg['emoji']}</span>
                <div>
                    <h4 style='margin:0;color:{cfg["color"]};'>{seg}</h4>
                    <span style='font-size:0.78rem;color:#888;'>
                        {count:,} customers · {pct:.1f}% of base
                    </span>
                </div>
            </div>
            <p><b>Profile:</b> {cfg['description']}</p>
            <p><b>Behaviour:</b> {cfg['behavior']}</p>
        </div>
        """, unsafe_allow_html=True)

        # RFM averages
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Recency",   f"{r_avg:.0f} days")
        m2.metric("Avg Frequency", f"{f_avg:.1f} orders")
        m3.metric("Avg Monetary",  f"£{m_avg:,.0f}")

        # Recommendations
        with st.expander("📋 Business recommendations", expanded=False):
            for rec in cfg["recommendations"]:
                st.markdown(f"- {rec}")

        st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE — VISUALISATIONS
# ─────────────────────────────────────────────
def page_viz(df: pd.DataFrame):
    st.markdown('<p class="section-label">Visualisations</p>',
                unsafe_allow_html=True)
    st.markdown('<h1 class="display-heading">Cluster analysis.</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="display-sub">Explore the structure of your customer '
                'segments across RFM dimensions.</p>',
                unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="section-label">Cluster distribution</p>',
                    unsafe_allow_html=True)
        fig = chart_distribution(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        st.markdown('<p class="section-label">Frequency vs Monetary</p>',
                    unsafe_allow_html=True)
        fig = chart_scatter(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    st.markdown('<p class="section-label">PCA — 2D projection of RFM space</p>',
                unsafe_allow_html=True)
    st.caption("Principal Component Analysis reduces the three RFM dimensions to two "
               "so we can visualise cluster separation.")
    try:
        fig = chart_pca(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install scikit-learn to render the PCA chart.")

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # RFM summary table
    st.markdown('<p class="section-label">Segment summary statistics</p>',
                unsafe_allow_html=True)
    summary = (
        df.groupby("Segment")
          .agg(
              Customers=("Recency", "count"),
              Recency_avg=("Recency", "mean"),
              Frequency_avg=("Frequency", "mean"),
              Monetary_avg=("Monetary", "mean"),
          )
          .round(1)
          .rename(columns={
              "Recency_avg": "Avg Recency (days)",
              "Frequency_avg": "Avg Frequency",
              "Monetary_avg": "Avg Monetary (£)",
          })
    )
    st.dataframe(summary, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE — BATCH PREDICT
# ─────────────────────────────────────────────
def page_batch(kmeans, scaler, label_map):
    st.markdown('<p class="section-label">Batch Predict</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="display-heading">Segment your list.</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="display-sub">Upload a CSV with customer RFM values '
                'and download the results with segments assigned.</p>',
                unsafe_allow_html=True)
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # ── Instructions ──
    st.markdown('<p class="section-label">Required format</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-block">
        <h4>CSV must contain these three columns (headers are case-sensitive):</h4>
        <p><code>Recency</code> — days since last purchase (integer)<br>
           <code>Frequency</code> — number of orders (integer)<br>
           <code>Monetary</code> — total spend in £ (decimal)</p>
        <p>Any additional columns (e.g. Customer ID, Name) will be kept and passed through.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sample download ──
    sample_df = pd.DataFrame({
        "Customer ID": ["C001", "C002", "C003", "C004", "C005"],
        "Recency":     [15,     90,     200,    45,     310],
        "Frequency":   [14,     6,      3,      2,      1],
        "Monetary":    [2100,   450,    280,    130,    60],
    })
    st.download_button(
        "⬇️ Download sample CSV template",
        data=sample_df.to_csv(index=False).encode(),
        file_name="rfm_template.csv",
        mime="text/csv",
    )

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

    # ── File uploader ──
    st.markdown('<p class="section-label">Upload your file</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        st.markdown("""
        <div style='padding:2.5rem 1rem;text-align:center;color:#bbb;'>
            <div style='font-size:2.5rem;'>📂</div>
            <p style='font-size:0.9rem;margin-top:0.8rem;'>
                Upload a CSV above to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return

    try:
        df_in = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    # ── Validate columns ──
    required = {"Recency", "Frequency", "Monetary"}
    missing_cols = required - set(df_in.columns)
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return

    # ── Check for nulls ──
    null_counts = df_in[["Recency", "Frequency", "Monetary"]].isnull().sum()
    if null_counts.any():
        st.warning(f"⚠️ Rows with missing RFM values will be skipped: "
                   f"{null_counts[null_counts > 0].to_dict()}")
        df_in = df_in.dropna(subset=["Recency", "Frequency", "Monetary"])

    if df_in.empty:
        st.error("No valid rows remaining after removing nulls.")
        return

    # ── Preview ──
    st.markdown('<p class="section-label">Preview (first 5 rows)</p>',
                unsafe_allow_html=True)
    st.dataframe(df_in.head(), use_container_width=True)

    # ── Run predictions ──
    with st.spinner("Segmenting customers…"):
        rfm_vals = df_in[["Recency", "Frequency", "Monetary"]].copy()
        log_vals = np.log1p(rfm_vals)
        log_vals.columns = ["Recency_log", "Frequency_log", "Monetary_log"]
        scaled = scaler.transform(log_vals)
        cluster_ids = kmeans.predict(scaled)
        segments = [label_map.get(int(c), f"Cluster {c}") for c in cluster_ids]

    df_out = df_in.copy()
    df_out["Cluster"]  = cluster_ids
    df_out["Segment"]  = segments

    # ── Results summary ──
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Segment breakdown</p>', unsafe_allow_html=True)

    seg_counts = df_out["Segment"].value_counts()
    total = len(df_out)
    cols = st.columns(len(SEGMENT_CONFIG))
    for col, (seg, cfg) in zip(cols, SEGMENT_CONFIG.items()):
        count = seg_counts.get(seg, 0)
        pct   = count / total * 100
        col.markdown(f"""
        <div class="stat-card" style="border-left:5px solid {cfg['color']};text-align:center;">
            <div style='font-size:1.4rem;'>{cfg['emoji']}</div>
            <h3 style='font-size:1.3rem;color:{cfg['color']};margin:4px 0;'>{count:,}</h3>
            <p style='font-size:0.78rem;'>{seg}<br>
               <span style='color:#aaa;'>{pct:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Full results table ──
    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Full results</p>', unsafe_allow_html=True)
    st.dataframe(df_out, use_container_width=True)

    # ── Download ──
    csv_bytes = df_out.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download segmented CSV",
        data=csv_bytes,
        file_name="customer_segments_batch.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Load models
    kmeans, scaler, label_map, err = load_models()

    # Sidebar navigation
    page = sidebar_nav()

    # Demo data (always available for charts)
    df_demo = generate_demo_rfm()

    # Model error banner (non-blocking for overview / insights / viz)
    if err:
        st.warning(
            f"⚠️ Could not load model files: `{err}` — "
            "place `kmeans_model.joblib`, `scaler.joblib`, and "
            "`segment_label_map.joblib` in the `models/` folder. "
            "Charts and insights are showing demo data.",
            icon="⚠️",
        )

    # Route pages
    if page == "🏠  Overview":
        page_overview(df_demo)

    elif page == "🔍  Predict Segment":
        if kmeans is None:
            st.error("❌ Model files not found. Cannot run predictions.")
        else:
            page_predict(kmeans, scaler, label_map, df_demo)

    elif page == "📂  Batch Predict":
        if kmeans is None:
            st.error("❌ Model files not found. Cannot run predictions.")
        else:
            page_batch(kmeans, scaler, label_map)

    elif page == "📊  Cluster Insights":
        page_insights(df_demo)

    elif page == "📈  Visualisations":
        page_viz(df_demo)


if __name__ == "__main__":
    main()
