import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

# =====================================================
# CUSTOM CSS — LIGHT, PROFESSIONAL THEME
# =====================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp{
    background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#FFFFFF;
    border-right:1px solid #E2E8F0;
}

section[data-testid="stSidebar"] *{
    color:#1E293B;
}

section[data-testid="stSidebar"] .stRadio label{
    font-weight:500;
}

/* Header banner */
.banner{
    background: linear-gradient(120deg, #0F172A 0%, #1E3A8A 55%, #0EA5E9 100%);
    border-radius:18px;
    padding:32px 36px;
    margin-bottom:28px;
    box-shadow: 0 10px 25px -8px rgba(15,23,42,0.35);
}

.banner-title{
    font-size:34px;
    font-weight:800;
    color:#FFFFFF;
    margin:0;
    letter-spacing:-0.5px;
}

.banner-subtitle{
    color:#CBD5E1;
    font-size:16px;
    margin-top:6px;
    font-weight:400;
}

.badge{
    display:inline-block;
    background:rgba(255,255,255,0.12);
    color:#E0F2FE;
    border:1px solid rgba(255,255,255,0.25);
    border-radius:999px;
    padding:5px 14px;
    font-size:12.5px;
    font-weight:600;
    margin-top:14px;
    margin-right:8px;
    letter-spacing:0.3px;
}

/* Section titles */
.section-title{
    font-size:20px;
    font-weight:700;
    color:#0F172A;
    margin:6px 0 14px 0;
    border-left:4px solid #0EA5E9;
    padding-left:10px;
}

/* Card wrapper for tables / content blocks */
.card{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:20px 22px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    margin-bottom:22px;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:#FFFFFF;
    border-radius:14px;
    border:1px solid #E2E8F0;
    padding:20px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}

div[data-testid="metric-container"] label{
    color:#64748B !important;
    font-weight:600;
}

div[data-testid="metric-container"] div{
    color:#0F172A;
}

/* Buttons */
.stButton>button{
    width:100%;
    background:#0EA5E9;
    color:white;
    border:none;
    border-radius:10px;
    height:46px;
    font-weight:600;
    transition: all 0.15s ease-in-out;
}

.stButton>button:hover{
    background:#0284C7;
    box-shadow:0 4px 12px rgba(2,132,199,0.35);
}

.stDownloadButton>button{
    width:100%;
    background:#16A34A;
    color:white;
    border:none;
    border-radius:10px;
    height:46px;
    font-weight:600;
}

.stDownloadButton>button:hover{
    background:#15803D;
}

/* Dataframe */
[data-testid="stDataFrame"]{
    border-radius:12px;
    overflow:hidden;
    border:1px solid #E2E8F0;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"]{
    background:#FFFFFF;
    border:1.5px dashed #94A3B8;
    border-radius:12px;
}

/* Divider look */
hr{
    border:none;
    border-top:1px solid #E2E8F0;
    margin:22px 0;
}

/* Footer */
.footer{
    text-align:center;
    color:#94A3B8;
    font-size:13.5px;
    padding-top:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown(
    "<div style='font-size:22px;font-weight:800;color:#0F172A;'>🛡️ NIDS Dashboard</div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div style='color:#64748B;font-size:13px;margin-bottom:16px;'>Traffic Classification Console</div>",
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 Prediction",
        "📊 Statistics",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("Model : Random Forest")
st.sidebar.info("Dataset : NSL-KDD")

# =====================================================
# SHARED BANNER
# =====================================================

def render_banner(title, subtitle):
    st.markdown(f"""
    <div class="banner">
        <p class="banner-title">{title}</p>
        <p class="banner-subtitle">{subtitle}</p>
        <span class="badge">🌲 Random Forest</span>
        <span class="badge">📂 NSL-KDD</span>
        <span class="badge">⚡ Real-Time Classification</span>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "🏠 Dashboard":

    render_banner(
        "🛡️ Network Intrusion Detection System",
        "Machine Learning Based Network Attack Detection Dashboard"
    )

    uploaded_file = st.file_uploader(
        "Upload cleaned_dataset.csv",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        total_flows = len(df)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flows", total_flows)
        col2.metric("Normal Flows", "-")
        col3.metric("Attack Flows", "-")
        col4.metric("Model", "Random Forest")

        st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.info("Go to the Prediction page from the sidebar to classify traffic.")

    else:
        st.warning("Upload cleaned_dataset.csv to get started.")

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "🔍 Prediction":

    render_banner(
        "🔍 Network Traffic Prediction",
        "Upload processed traffic data and classify it in real time"
    )

    uploaded_file = st.file_uploader(
        "Upload Processed CSV File",
        type=["csv"],
        key="prediction_file"
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully")

        st.markdown('<p class="section-title">Dataset Preview</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Predict Network Traffic"):

            with st.spinner("Running Random Forest model..."):

                if "label" in df.columns:
                    X = df.drop("label", axis=1)
                else:
                    X = df.copy()

                predictions = model.predict(X)
                prediction_labels = label_encoder.inverse_transform(predictions)

                result = df.copy()
                result["Prediction"] = prediction_labels

                st.success("Prediction completed successfully")

                st.markdown('<p class="section-title">Prediction Results</p>', unsafe_allow_html=True)
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                csv = result.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "📥 Download Prediction Results",
                    csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

                st.session_state["prediction_result"] = result
                st.session_state["prediction_labels"] = prediction_labels

                prediction_series = pd.Series(prediction_labels)
                normal_count = (prediction_series == "normal").sum()
                attack_count = len(prediction_series) - normal_count

                st.markdown('<p class="section-title">Prediction Summary</p>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", len(result))
                col2.metric("Normal", normal_count)
                col3.metric("Attack", attack_count)
                col4.metric("Model", "Random Forest")

    else:
        st.info("Please upload cleaned_dataset.csv to run a prediction.")

# =====================================================
# STATISTICS PAGE
# =====================================================

elif page == "📊 Statistics":

    render_banner(
        "📊 Attack Statistics Dashboard",
        "A breakdown of classified traffic across all attack categories"
    )

    if "prediction_labels" not in st.session_state:

        st.warning("Please run a prediction first from the Prediction page.")

    else:

        prediction_series = pd.Series(st.session_state["prediction_labels"])
        result = st.session_state["prediction_result"]

        st.markdown('<p class="section-title">Attack Frequency</p>', unsafe_allow_html=True)

        attack_stats = prediction_series.value_counts()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(
            attack_stats.rename_axis("Attack Type").reset_index(name="Count"),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p class="section-title">Pie Chart</p>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            fig.patch.set_facecolor("#FFFFFF")
            colors = plt.cm.Blues_r(
                [i / max(len(attack_stats) - 1, 1) for i in range(len(attack_stats))]
            )
            ax.pie(
                attack_stats.values,
                labels=attack_stats.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                textprops={"color": "#0F172A"}
            )
            ax.axis("equal")
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<p class="section-title">Bar Chart</p>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)

            chart_df = pd.DataFrame({
                "Attack Type": attack_stats.index,
                "Count": attack_stats.values
            })
            st.bar_chart(chart_df.set_index("Attack Type"), color="#0EA5E9")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-title">Prediction Sample</p>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.dataframe(result.head(20), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =====================================================
# ABOUT PAGE
# =====================================================

elif page == "ℹ️ About":

    render_banner(
        "ℹ️ About This Project",
        "Details on the dataset, model, and technologies behind this system"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        This project detects malicious network traffic using a machine
        learning classifier trained on labeled flow data. It provides
        an interactive interface for uploading traffic samples, running
        predictions, and reviewing attack statistics.
        <br><br>
        <b>Dataset:</b> NSL-KDD<br>
        <b>Model:</b> Random Forest Classifier
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Technologies</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        Python &bull; Streamlit &bull; Pandas &bull; Matplotlib &bull; Joblib &bull; Scikit-Learn
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">Responsibilities</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        ✔ Design the Streamlit dashboard<br>
        ✔ Connect the trained ML model<br>
        ✔ Enable CSV upload<br>
        ✔ Generate prediction results<br>
        ✔ Compute attack statistics<br>
        ✔ Build charts and visualizations
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title">Features</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        ✔ Upload processed datasets<br>
        ✔ Detect network attacks<br>
        ✔ View prediction results<br>
        ✔ Pie and bar chart visualizations<br>
        ✔ Download predictions as CSV
        </div>
        """, unsafe_allow_html=True)

    st.success("Network Intrusion Detection System Dashboard")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
"""
<div class="footer">
Network Intrusion Detection System<br>
Developed using Streamlit • Random Forest • NSL-KDD Dataset
</div>
""",
unsafe_allow_html=True
)