"""Gradio deployment app for the saved XGBoost churn model.

Run from this folder with: python app.py
"""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "xgbclassifier.joblib"

# These match the column names and order stored with xgbclassifier.joblib.
FEATURE_COLUMNS = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

MODEL = joblib.load(MODEL_PATH)


def predict_churn(
    account_weeks: int,
    contract_renewal: int,
    data_plan: int,
    data_usage: float,
    customer_service_calls: int,
    day_minutes: float,
    day_calls: int,
    monthly_charge: float,
    overage_fee: float,
    roaming_minutes: float,
) -> tuple[str, float]:
    """Predict whether one customer is likely to churn."""
    customer = pd.DataFrame(
        [[
            account_weeks,
            contract_renewal,
            data_plan,
            data_usage,
            customer_service_calls,
            day_minutes,
            day_calls,
            monthly_charge,
            overage_fee,
            roaming_minutes,
        ]],
        columns=FEATURE_COLUMNS,
    )

    churn_probability = float(MODEL.predict_proba(customer)[0, 1])
    prediction = "Likely to churn" if churn_probability >= 0.5 else "Unlikely to churn"
    return prediction, round(churn_probability, 4)


with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown(
        "# Customer Churn Prediction\n"
        "Enter a telecom customer's details to predict churn with the saved XGBoost model."
    )

    with gr.Row():
        with gr.Column():
            account_weeks = gr.Slider(0, 300, value=100, step=1, label="Account weeks")
            contract_renewal = gr.Radio([0, 1], value=1, label="Contract renewal (1 = Yes)")
            data_plan = gr.Radio([0, 1], value=0, label="Data plan (1 = Yes)")
            data_usage = gr.Number(value=2.0, minimum=0, label="Data usage")
            customer_service_calls = gr.Slider(0, 15, value=2, step=1, label="Customer-service calls")
        with gr.Column():
            day_minutes = gr.Number(value=180.0, minimum=0, label="Day minutes")
            day_calls = gr.Slider(0, 200, value=100, step=1, label="Day calls")
            monthly_charge = gr.Number(value=60.0, minimum=0, label="Monthly charge")
            overage_fee = gr.Number(value=5.0, minimum=0, label="Overage fee")
            roaming_minutes = gr.Number(value=10.0, minimum=0, label="Roaming minutes")

    predict_button = gr.Button("Predict churn", variant="primary")
    prediction = gr.Label(label="Prediction")
    probability = gr.Number(label="Churn probability", precision=4)

    predict_button.click(
        fn=predict_churn,
        inputs=[
            account_weeks,
            contract_renewal,
            data_plan,
            data_usage,
            customer_service_calls,
            day_minutes,
            day_calls,
            monthly_charge,
            overage_fee,
            roaming_minutes,
        ],
        outputs=[prediction, probability],
    )


if __name__ == "__main__":
    demo.launch(share=True)
