import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# -------------------- RANDOM FOREST --------------------------

RF_MODEL_PATH = (
    BASE_DIR
    / "final_rf_16_devices_60s_2000trees.joblib"
)

RF_SCHEMA_PATH = (
    BASE_DIR
    / "rf_feature_schema.json"
)

DATA_PATH = (
    BASE_DIR
    / "rf_ready_16_devices_fixed.parquet"
)

# -------------------- XGBOOST -------------------------------

XGB_MODEL_PATH = (
    BASE_DIR
    / "final_xgb_5class_60s_1500trees.joblib"
)

XGB_SCHEMA_PATH = (
    BASE_DIR
    / "xgb_feature_schema.json"
)

XGB_MAPPING_PATH = (
    BASE_DIR
    / "xgb_class_mapping.json"
)

# ============================================================
# LOAD MODELS / DATA
# ============================================================

@st.cache_resource
def load_rf_model():
    return joblib.load(RF_MODEL_PATH)


@st.cache_resource
def load_xgb_model():
    return joblib.load(XGB_MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_rf_schema():
    with open(RF_SCHEMA_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_xgb_schema():
    with open(XGB_SCHEMA_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_xgb_mapping():
    with open(XGB_MAPPING_PATH, "r") as f:
        return json.load(f)


# ============================================================
# LOAD EVERYTHING
# ============================================================

rf_model = load_rf_model()
xgb_model = load_xgb_model()

df = load_data()

rf_feature_cols = load_rf_schema()
xgb_feature_cols = load_xgb_schema()
xgb_class_mapping = load_xgb_mapping()


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="IoT Behavioral Fingerprinting",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ IoT Behavioral Fingerprinting")

st.caption(
    "Two-stage behavioral detection using Random Forest + XGBoost"
)


# ============================================================
# SESSION STATE
# ============================================================

if "rf_result" not in st.session_state:
    st.session_state.rf_result = None

if "xgb_result" not in st.session_state:
    st.session_state.xgb_result = None

if "prediction_context" not in st.session_state:
    st.session_state.prediction_context = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Traffic Replay")

devices = sorted(df["device"].unique())

selected_device = st.sidebar.selectbox(
    "Select IoT Device",
    devices
)

device_df = (
    df[df["device"] == selected_device]
    .reset_index(drop=True)
)

window_number = st.sidebar.number_input(
    "Select Traffic Window",
    min_value=1,
    max_value=len(device_df),
    value=1,
    step=1
)


# ============================================================
# CURRENT WINDOW CONTEXT
# ============================================================

current_context = (
    str(selected_device),
    int(window_number)
)


# ============================================================
# RF PREDICT BUTTON
# ============================================================

predict_clicked = st.sidebar.button(
    "🔍 Predict This Window",
    type="primary",
    use_container_width=True
)


# ============================================================
# SELECTED WINDOW
# ============================================================

selected_row = device_df.iloc[window_number - 1]


# ============================================================
# RESET OLD RESULTS WHEN WINDOW / DEVICE CHANGES
# ============================================================

if (
    st.session_state.prediction_context is not None
    and st.session_state.prediction_context != current_context
    and not predict_clicked
):
    st.session_state.rf_result = None
    st.session_state.xgb_result = None
    st.session_state.prediction_context = None


# ============================================================
# BASIC TRAFFIC INFORMATION
# ============================================================

st.subheader("Traffic Input")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Device",
        selected_device
    )

with col2:
    st.metric(
        "Traffic Window",
        f"{window_number} / {len(device_df)}"
    )

with col3:
    st.metric(
        "Input Features",
        "44"
    )


# ============================================================
# RF PREDICTION
# ============================================================

if predict_clicked:

    # --------------------------------------------------------
    # ONLY 44 FEATURES ENTER RANDOM FOREST
    # --------------------------------------------------------

    X_input = pd.DataFrame(
        [{
            col: selected_row[col]
            for col in rf_feature_cols
        }]
    )

    # Make sure all RF inputs are numeric.
    # Missing values remain NaN.
    X_input = X_input.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # RF PREDICTION
    # --------------------------------------------------------

    rf_prediction = int(
        rf_model.predict(X_input)[0]
    )

    rf_probabilities = (
        rf_model.predict_proba(X_input)[0]
    )

    # --------------------------------------------------------
    # SAVE RF RESULT
    # --------------------------------------------------------

    st.session_state.rf_result = {
        "prediction": rf_prediction,
        "benign_probability": float(
            rf_probabilities[0]
        ),
        "malicious_probability": float(
            rf_probabilities[1]
        )
    }

    st.session_state.prediction_context = (
        current_context
    )

    # New RF prediction means a previous
    # XGB result must be cleared.
    st.session_state.xgb_result = None


# ============================================================
# BEFORE ANY PREDICTION
# ============================================================

if (
    st.session_state.rf_result is None
    or st.session_state.prediction_context
    != current_context
):

    st.divider()

    st.info(
        "Select a device and traffic window, then click "
        "**Predict This Window** to run the Random Forest."
    )

    st.subheader("44-Feature Behavioral Input")

    X_preview = pd.DataFrame(
        [{
            col: selected_row[col]
            for col in rf_feature_cols
        }]
    ).T

    X_preview.columns = ["value"]

    st.dataframe(
        X_preview,
        use_container_width=True
    )


# ============================================================
# DISPLAY RF RESULT
# ============================================================

if (
    st.session_state.rf_result is not None
    and st.session_state.prediction_context
    == current_context
):

    rf_result = st.session_state.rf_result

    rf_prediction = rf_result["prediction"]

    benign_probability = (
        rf_result["benign_probability"]
    )

    malicious_probability = (
        rf_result["malicious_probability"]
    )


    # ========================================================
    # RF PREDICTION
    # ========================================================

    st.divider()

    st.subheader("RF Prediction")

    if rf_prediction == 1:

        st.error(
            "🚨 MALICIOUS BEHAVIOR DETECTED"
        )

    else:

        st.success(
            "✅ BENIGN BEHAVIOR DETECTED"
        )


    # ========================================================
    # RF PROBABILITIES
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Benign Probability",
            f"{benign_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Malicious Probability",
            f"{malicious_probability * 100:.2f}%"
        )


    # ========================================================
    # RF GROUND TRUTH
    # ========================================================

    st.divider()

    st.subheader(
        "RF Ground Truth Comparison"
    )

    # Ground truth is accessed only after RF prediction.
    # It is NOT passed to the RF.

    actual = int(
        selected_row["is_malicious"]
    )

    actual_label = (
        "MALICIOUS"
        if actual == 1
        else "BENIGN"
    )

    predicted_label = (
        "MALICIOUS"
        if rf_prediction == 1
        else "BENIGN"
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "RF Prediction",
            predicted_label
        )

    with col2:

        st.metric(
            "IoT-23 Ground Truth",
            actual_label
        )

    with col3:

        if rf_prediction == actual:

            st.success("✓ CORRECT")

        else:

            st.error("✗ INCORRECT")


    # ========================================================
    # RF FEATURES
    # ========================================================

    st.divider()

    st.subheader(
        "44 Features Given to the Random Forest"
    )

    X_input_display = pd.DataFrame(
        [{
            col: selected_row[col]
            for col in rf_feature_cols
        }]
    ).T

    X_input_display.columns = ["value"]

    st.dataframe(
        X_input_display,
        use_container_width=True
    )


    # ========================================================
    # XGBOOST STAGE
    # ========================================================

    if rf_prediction == 1:

        st.divider()

        st.subheader(
            "Attack Category Classification"
        )

        st.info(
            "Random Forest detected malicious behavior. "
            "Run XGBoost to classify the attack category."
        )


        # ----------------------------------------------------
        # XGBOOST BUTTON
        # ----------------------------------------------------

        xgb_clicked = st.button(
            "🎯 Predict Attack Category",
            type="primary"
        )


        # ----------------------------------------------------
        # RUN XGBOOST
        # ----------------------------------------------------

        if xgb_clicked:

            # ================================================
            # VERIFY FEATURE SCHEMAS
            # ================================================

            if rf_feature_cols != xgb_feature_cols:

                st.error(
                    "RF and XGBoost feature schemas do not match."
                )

                st.stop()


            # ================================================
            # BUILD XGBOOST INPUT
            # ================================================
            #
            # IMPORTANT:
            #
            # We explicitly construct a new DataFrame
            # instead of using:
            #
            # selected_row[xgb_feature_cols]
            #
            # directly.
            #
            # This prevents pandas from converting the
            # entire single row into object dtype because
            # the original row also contains metadata.
            # ================================================

            X_xgb = pd.DataFrame(
                [{
                    col: selected_row[col]
                    for col in xgb_feature_cols
                }]
            )

            # Convert ONLY the 44 model features to numeric.
            #
            # Valid numeric values remain numeric.
            # Missing values become NaN.
            # NaN is NOT converted to zero.

            X_xgb = X_xgb.apply(
                pd.to_numeric,
                errors="coerce"
            )


            # ================================================
            # XGBOOST PREDICTION
            # ================================================

            xgb_prediction_encoded = int(
                xgb_model.predict(X_xgb)[0]
            )

            xgb_probabilities = (
                xgb_model.predict_proba(X_xgb)[0]
            )


            # ================================================
            # CLASS MAPPING
            # ================================================

            predicted_attack = xgb_class_mapping[
                str(xgb_prediction_encoded)
            ]


            # ================================================
            # XGBOOST CONFIDENCE
            # ================================================

            xgb_confidence = float(
                xgb_probabilities[
                    xgb_prediction_encoded
                ]
            )


            # ================================================
            # SAVE XGBOOST RESULT
            # ================================================

            st.session_state.xgb_result = {

                "prediction_encoded":
                    xgb_prediction_encoded,

                "prediction":
                    predicted_attack,

                "confidence":
                    xgb_confidence,

                "probabilities":
                    xgb_probabilities.tolist()
            }


    # ========================================================
    # DISPLAY XGBOOST RESULT
    # ========================================================

    if st.session_state.xgb_result is not None:

        xgb_result = (
            st.session_state.xgb_result
        )

        predicted_attack = (
            xgb_result["prediction"]
        )

        xgb_confidence = (
            xgb_result["confidence"]
        )

        xgb_probabilities = (
            xgb_result["probabilities"]
        )


        # ====================================================
        # XGBOOST PREDICTION
        # ====================================================

        st.subheader(
            "XGBoost Prediction"
        )

        st.warning(
            f"🎯 **{predicted_attack}**"
        )

        st.metric(
            "XGBoost Confidence",
            f"{xgb_confidence * 100:.2f}%"
        )


        # ====================================================
        # XGBOOST GROUND TRUTH
        # ====================================================

        st.divider()

        st.subheader(
            "XGBoost Ground Truth Comparison"
        )

        actual_attack = (
            selected_row["attack_category"]
        )

        if pd.isna(actual_attack):

            actual_attack = "UNKNOWN"

        else:

            actual_attack = str(
                actual_attack
            )


        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "XGBoost Prediction",
                predicted_attack
            )

        with col2:

            st.metric(
                "IoT-23 Ground Truth",
                actual_attack
            )

        with col3:

            if predicted_attack == actual_attack:

                st.success("✓ CORRECT")

            else:

                st.error("✗ INCORRECT")


        # ====================================================
        # XGBOOST FEATURES
        # ====================================================

        st.divider()

        st.subheader(
            "44 Features Given to XGBoost"
        )

        # Rebuild numerically typed XGB input
        # for display as well.

        X_xgb_display = pd.DataFrame(
            [{
                col: selected_row[col]
                for col in xgb_feature_cols
            }]
        ).T

        X_xgb_display.columns = ["value"]

        st.dataframe(
            X_xgb_display,
            use_container_width=True
        )


        # ====================================================
        # XGBOOST CLASS PROBABILITIES
        # ====================================================

        st.divider()

        st.subheader(
            "XGBoost Attack Category Probabilities"
        )

        probability_data = []

        for class_id, probability in enumerate(
            xgb_probabilities
        ):

            class_name = xgb_class_mapping[
                str(class_id)
            ]

            probability_data.append(
                {
                    "Attack Category": class_name,
                    "Probability": (
                        f"{probability * 100:.2f}%"
                    )
                }
            )


        probability_df = pd.DataFrame(
            probability_data
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # INFERENCE FLOW
    # ========================================================

    st.divider()

    st.subheader("Inference Flow")

    if rf_prediction == 1:

        if st.session_state.xgb_result is not None:

            st.code(
                """44 behavioral features
        ↓
Random Forest (2000 trees)
        ↓
MALICIOUS
        ↓
XGBoost (1500 trees)
        ↓
5 Attack Categories
        ↓
Attack Category Prediction
""",
                language="text"
            )

        else:

            st.code(
                """44 behavioral features
        ↓
Random Forest (2000 trees)
        ↓
MALICIOUS
        ↓
[ Predict Attack Category ]
        ↓
XGBoost
""",
                language="text"
            )

    else:

        st.code(
            """44 behavioral features
        ↓
Random Forest (2000 trees)
        ↓
BENIGN
        ↓
Inference stops
""",
            language="text"
        )
