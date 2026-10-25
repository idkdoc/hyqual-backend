
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime


# ==================================================
# MODEL PATHS
# ==================================================

MODEL_DIR = Path(__file__).resolve().parent / "models"

RF_PATH = MODEL_DIR / "random_forest_risk_model.joblib"

ARIMA_DIR = MODEL_DIR


# ==================================================
# WATER QUALITY THRESHOLDS
# ==================================================

# These are the thresholds from your existing code.
# Verify their regulatory and aquaculture references
# before presenting them as official DAO requirements.

PARAMETERS = {
    "DO (mg/L)": {
        "healthy": lambda x: x >= 5.0,
        "conditional": lambda x: 3.7 <= x < 5.0,
        "unit": "mg/L"
    },

    "pH": {
        "healthy": lambda x: 7.5 <= x <= 8.5,
        "conditional": lambda x: (
            6.5 <= x < 7.5 or 8.5 < x <= 9.0
        ),
        "unit": ""
    },

    "Temp (OC)": {
        "healthy": lambda x: 26.0 <= x <= 30.0,
        "conditional": lambda x: (
            25.0 <= x < 26.0 or 30.0 < x <= 31.0
        ),
        "unit": "°C"
    },

    "Salinity (ppt)": {
        "healthy": lambda x: 15.0 <= x <= 25.0,
        "conditional": lambda x: (
            10.0 <= x < 15.0 or 25.0 < x <= 30.0
        ),
        "unit": "ppt"
    }
}


STATUS_NAMES = {
    0: "Healthy",
    1: "Conditional",
    2: "Unhealthy"
}


# ==================================================
# VALIDATE SENSOR READINGS
# ==================================================

def validate_readings(do, ph, temperature, salinity):

    readings = {
        "DO (mg/L)": do,
        "pH": ph,
        "Temp (OC)": temperature,
        "Salinity (ppt)": salinity
    }

    for name, value in readings.items():

        if not isinstance(value, (int, float, np.number)):
            raise ValueError(
                f"{name} must be a numeric value"
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be a finite number"
            )

    if do < 0:
        raise ValueError("DO cannot be negative")

    if not 0 <= ph <= 14:
        raise ValueError("pH must be between 0 and 14")

    if salinity < 0:
        raise ValueError("Salinity cannot be negative")

    if temperature < -273.15:
        raise ValueError("Invalid temperature")


# ==================================================
# DAO-STYLE WATER QUALITY ASSESSMENT
# ==================================================

def parameter_status(value, parameter):

    rules = PARAMETERS[parameter]

    if rules["healthy"](value):
        return 0

    if rules["conditional"](value):
        return 1

    return 2


def assess_water_quality(do, ph, temperature, salinity):

    validate_readings(
        do,
        ph,
        temperature,
        salinity
    )

    values = {
        "DO (mg/L)": do,
        "pH": ph,
        "Temp (OC)": temperature,
        "Salinity (ppt)": salinity
    }

    statuses = {}

    for parameter, value in values.items():

        score = parameter_status(
            value,
            parameter
        )

        statuses[parameter] = {
            "value": float(value),
            "status": STATUS_NAMES[score],
            "score": score
        }

    if all(
        item["score"] == 0
        for item in statuses.values()
    ):
        condition = "Normal"

    else:
        condition = "Abnormal"

    return {
        "condition": condition,
        "parameters": statuses
    }


# ==================================================
# LOAD RANDOM FOREST MODEL
# ==================================================

if not RF_PATH.is_file():
    raise FileNotFoundError(
        f"Random Forest model not found: {RF_PATH}"
    )

rf_bundle = joblib.load(RF_PATH)

rf_model = rf_bundle["model"]
rf_features = rf_bundle["features"]


# ==================================================
# LOAD FOUR ARIMA MODELS
# ==================================================

arima_models = {}

for path in ARIMA_DIR.glob("*.joblib"):

    if path == RF_PATH:
        continue

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        continue

    if "model" not in bundle:
        continue

    parameter = bundle.get("parameter")

    if parameter in PARAMETERS:
        arima_models[parameter] = bundle


missing_models = [
    parameter
    for parameter in PARAMETERS
    if parameter not in arima_models
]

if missing_models:
    raise FileNotFoundError(
        "Missing ARIMA models for: "
        + ", ".join(missing_models)
    )


# ==================================================
# BUILD RANDOM FOREST INPUT
# ==================================================

def build_rf_input(
    do,
    ph,
    temperature,
    salinity,
    timestamp
):

    if not isinstance(timestamp, datetime):
        raise ValueError(
            "timestamp must be a datetime object"
        )

    hour = timestamp.hour + timestamp.minute / 60
    month = timestamp.month

    row = {
        "DO (mg/L)": do,
        "pH": ph,
        "Temp (OC)": temperature,
        "Salinity (ppt)": salinity,

        "hour_sin": np.sin(
            2 * np.pi * hour / 24
        ),

        "hour_cos": np.cos(
            2 * np.pi * hour / 24
        ),

        "month_sin": np.sin(
            2 * np.pi * month / 12
        ),

        "month_cos": np.cos(
            2 * np.pi * month / 12
        )
    }

    return pd.DataFrame([row])[rf_features]


# ==================================================
# RANDOM FOREST RISK PREDICTION
# ==================================================

def predict_risk(
    do,
    ph,
    temperature,
    salinity,
    timestamp
):

    model_input = build_rf_input(
        do,
        ph,
        temperature,
        salinity,
        timestamp
    )

    prediction = str(
        rf_model.predict(model_input)[0]
    )

    if prediction not in ["Low", "Moderate", "High"]:
        raise ValueError(
            f"Unexpected RF prediction: {prediction}"
        )

    probabilities = {}

    if hasattr(rf_model, "predict_proba"):

        probs = rf_model.predict_proba(
            model_input
        )[0]

        for label, probability in zip(
            rf_model.classes_,
            probs
        ):

            probabilities[str(label)] = float(
                probability
            )

    return {
        "prediction": prediction,
        "probabilities": probabilities
    }


# ==================================================
# ARIMA FORECASTING
# ==================================================

def forecast_water_quality(periods=7):

    if not isinstance(periods, int) or periods < 2:
        raise ValueError(
            "Forecast periods must be an integer of at least 2"
        )

    result = pd.DataFrame({
        "Forecast Period": range(1, periods + 1)
    })

    for parameter in PARAMETERS:

        bundle = arima_models[parameter]

        model = bundle["model"]

        forecast = model.get_forecast(
            steps=periods
        )

        values = np.asarray(
            forecast.predicted_mean,
            dtype=float
        )

        if len(values) != periods:
            raise ValueError(
                f"Invalid forecast length for {parameter}"
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Invalid ARIMA forecast for {parameter}"
            )

        result[parameter] = values

    return result


# ==================================================
# ARIMA TREND CLASSIFICATION
# ==================================================

def classify_parameter_trend(values, parameter):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 2:
        raise ValueError(
            "At least two forecast values are required"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Forecast contains invalid values"
        )

    first_value = values[0]
    last_value = values[-1]

    change = last_value - first_value

    # --------------------------------------------------
    # DISSOLVED OXYGEN
    # --------------------------------------------------

    if parameter == "DO (mg/L)":

        tolerance = 0.25
        healthy_min = 5.0

        if abs(change) <= tolerance:
            return "Stable"

        if last_value < first_value:
            if last_value < healthy_min:
                return "Deteriorating"

            return "Stable"

        if last_value > first_value:
            return "Improving"

        return "Stable"

    # --------------------------------------------------
    # pH, TEMPERATURE, AND SALINITY
    # --------------------------------------------------

    ranges = {
        "pH": (7.5, 8.5, 0.10),
        "Temp (OC)": (26.0, 30.0, 0.25),
        "Salinity (ppt)": (15.0, 25.0, 0.50)
    }

    if parameter not in ranges:
        raise ValueError(
            f"Unknown parameter: {parameter}"
        )

    healthy_min, healthy_max, tolerance = ranges[
        parameter
    ]

    if abs(change) <= tolerance:
        return "Stable"

    # Distance outside the healthy range.
    # Values inside the range have distance 0.

    first_distance = max(
        healthy_min - first_value,
        0,
        first_value - healthy_max
    )

    last_distance = max(
        healthy_min - last_value,
        0,
        last_value - healthy_max
    )

    if last_distance < first_distance:
        return "Improving"

    if last_distance > first_distance:
        return "Deteriorating"

    return "Stable"


# ==================================================
# OVERALL ARIMA TREND
# ==================================================

def classify_overall_arima_trend(forecast_result):

    parameter_trends = {}

    for parameter in PARAMETERS:

        values = forecast_result[
            parameter
        ].values

        parameter_trends[
            parameter
        ] = classify_parameter_trend(
            values,
            parameter
        )

    trends = list(
        parameter_trends.values()
    )

    improving_count = trends.count(
        "Improving"
    )

    deteriorating_count = trends.count(
        "Deteriorating"
    )

    if deteriorating_count >= 3:
        overall_trend = "Deteriorating"

    elif improving_count >= 3:
        overall_trend = "Improving"

    elif deteriorating_count > improving_count:
        overall_trend = "Deteriorating"

    elif improving_count > deteriorating_count:
        overall_trend = "Improving"

    else:
        overall_trend = "Stable"

    return {
        "parameter_trends": parameter_trends,
        "overall_trend": overall_trend
    }


# ==================================================
# HYBRID EARLY WARNING
# ==================================================

def generate_hybrid_status(
    water_quality,
    rf_risk,
    arima_trend
):

    if water_quality not in ["Normal", "Abnormal"]:
        raise ValueError(
            "Invalid water quality condition"
        )

    if rf_risk not in ["Low", "Moderate", "High"]:
        raise ValueError(
            "Invalid Random Forest risk"
        )

    if arima_trend not in [
        "Stable",
        "Improving",
        "Deteriorating"
    ]:
        raise ValueError(
            "Invalid ARIMA trend"
        )

    # --------------------------------------------------
    # NORMAL CONDITION
    # --------------------------------------------------

    if water_quality == "Normal":

        if rf_risk == "Low":

            if arima_trend in [
                "Stable",
                "Improving"
            ]:
                return "Normal"

            return "Warning"

        if rf_risk == "Moderate":

            if arima_trend == "Deteriorating":
                return "Critical"

            return "Warning"

        if rf_risk == "High":

            if arima_trend == "Deteriorating":
                return "Critical"

            return "Warning"

    # --------------------------------------------------
    # ABNORMAL CONDITION
    # --------------------------------------------------

    if water_quality == "Abnormal":

        if rf_risk == "Low":
            return "Warning"

        if rf_risk == "Moderate":

            if arima_trend == "Deteriorating":
                return "Critical"

            return "Warning"

        if rf_risk == "High":

            if arima_trend == "Deteriorating":
                return "Critical"

            return "Warning"

    raise ValueError(
        "Unable to determine Hybrid status"
    )


# ==================================================
# COMPLETE HYBRID ANALYSIS
# ==================================================

def analyze_water_quality(
    do,
    ph,
    temperature,
    salinity,
    timestamp,
    periods=7
):

    # --------------------------------------------------
    # VALIDATE SENSOR INPUTS
    # --------------------------------------------------

    validate_readings(
        do,
        ph,
        temperature,
        salinity
    )

    # --------------------------------------------------
    # DAO ASSESSMENT
    # --------------------------------------------------

    water_quality = assess_water_quality(
        do,
        ph,
        temperature,
        salinity
    )

    # --------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------

    rf_result = predict_risk(
        do,
        ph,
        temperature,
        salinity,
        timestamp
    )

    # --------------------------------------------------
    # ARIMA FORECAST
    # --------------------------------------------------

    forecast_result = forecast_water_quality(
        periods
    )

    # --------------------------------------------------
    # ARIMA TREND
    # --------------------------------------------------

    arima_result = classify_overall_arima_trend(
        forecast_result
    )

    # --------------------------------------------------
    # HYBRID STATUS
    # --------------------------------------------------

    hybrid_status = generate_hybrid_status(
        water_quality["condition"],
        rf_result["prediction"],
        arima_result["overall_trend"]
    )

    return {
        "water_quality": water_quality,

        "random_forest": rf_result,

        "arima": {
            "forecasts": forecast_result,
            "parameter_trends": arima_result[
                "parameter_trends"
            ],
            "overall_trend": arima_result[
                "overall_trend"
            ]
        },

        "hybrid": {
            "status": hybrid_status
        }
    }


# ==================================================
# TEST HYQUAL ML BACKEND
# ==================================================

if __name__ == "__main__":

    print("HyQual ML Backend")
    print("------------------")

    # Sample sensor readings

    do = 5.0
    ph = 8.0
    temperature = 28.0
    salinity = 20.0

    timestamp = datetime(
        2026, 10, 1, 21, 2
    )

    # Run Hybrid analysis

    result = analyze_water_quality(
        do=do,
        ph=ph,
        temperature=temperature,
        salinity=salinity,
        timestamp=timestamp,
        periods=7
    )

    # DAO

    print("\nDAO Assessment:")
    print(
        "Condition:",
        result["water_quality"]["condition"]
    )

    for parameter, info in result[
        "water_quality"
    ]["parameters"].items():

        print(
            f"{parameter}: "
            f"{info['value']} -> "
            f"{info['status']}"
        )

    # Random Forest

    print("\nRandom Forest:")
    print(
        "Prediction:",
        result["random_forest"]["prediction"]
    )

    print("\nProbabilities:")

    for label, probability in result[
        "random_forest"
    ]["probabilities"].items():

        print(
            f"{label}: {probability * 100:.2f}%"
        )

    # ARIMA trends

    print("\nARIMA Parameter Trends:")

    for parameter, trend in result[
        "arima"
    ]["parameter_trends"].items():

        print(f"{parameter}: {trend}")

    print(
        "\nOverall ARIMA Trend:",
        result["arima"]["overall_trend"]
    )

    # ARIMA forecasts

    print("\nARIMA Forecasts:")

    forecast_df = result[
        "arima"
    ]["forecasts"]

    for _, row in forecast_df.iterrows():

        period = int(
            row["Forecast Period"]
        )

        print(f"\nPeriod {period}")

        for parameter in PARAMETERS:

            print(
                f"{parameter}: "
                f"{row[parameter]:.3f}"
            )

    # Hybrid final status

    print("\nHybrid Early Warning:")

    print(
        "Final Status:",
        result["hybrid"]["status"]
    )
