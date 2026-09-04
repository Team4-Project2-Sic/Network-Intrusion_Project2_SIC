import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Network Intrusion Detector", layout="wide")

ARTIFACT_DIR = "deployment_artifacts"

# Columns dropped before modeling (not used as features)
DROP_COLS = ["Label", "Binary_Label", "RST Flag Count", "Destination Port"]


@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{ARTIFACT_DIR}/xgb_model.pkl")
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
    imputer = joblib.load(f"{ARTIFACT_DIR}/imputer.pkl")
    with open(f"{ARTIFACT_DIR}/selected_features.json") as f:
        selected_features = json.load(f)
    with open(f"{ARTIFACT_DIR}/non_negative_cols.json") as f:
        non_negative_cols = json.load(f)
    return model, scaler, imputer, selected_features, non_negative_cols


def preprocess(df, imputer, scaler, selected_features, non_negative_cols):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # clean placeholders / infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.replace(["?", "unknown", "Unknown", "UNKNOWN", "None", "none", " "], np.nan, inplace=True)

    # drop target / excluded columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # mask invalid negative values the same way training did
    cols_present = [c for c in non_negative_cols if c in df.columns]
    df[cols_present] = df[cols_present].mask(df[cols_present] < 0)

    # numeric coercion, then impute the same columns the training imputer was fit on
    for c in cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols_present] = imputer.transform(df[cols_present])

    # any remaining gaps in other columns -> 0 (kept simple/direct)
    df = df.fillna(0)

    # align to the exact feature set/order the model was trained on
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df_selected = df[selected_features]

    X_scaled = scaler.transform(df_selected)
    return X_scaled


st.title("Network Intrusion Detector (XGBoost)")
st.caption("Upload raw network-flow CSV data. Preprocessing, feature selection, and scaling run automatically before prediction.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{raw_df.shape[0]}** rows, **{raw_df.shape[1]}** columns.")

    model, scaler, imputer, selected_features, non_negative_cols = load_artifacts()

    try:
        X_scaled = preprocess(raw_df, imputer, scaler, selected_features, non_negative_cols)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    result_df = raw_df.copy()
    result_df["Prediction"] = np.where(preds == 1, "ATTACK", "BENIGN")
    result_df["Attack_Probability"] = probs.round(4)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Flows", len(result_df))
    col2.metric("Predicted Attacks", int((preds == 1).sum()))
    col3.metric("Predicted Benign", int((preds == 0).sum()))

    st.bar_chart(result_df["Prediction"].value_counts())

    st.subheader("Results")
    st.dataframe(result_df, use_container_width=True)

    st.download_button(
        "Download results as CSV",
        data=result_df.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV with the raw network-flow columns to get predictions.")
