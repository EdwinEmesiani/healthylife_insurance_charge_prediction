# app.py

import json
import uuid
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import CommitScheduler

# 1. Load trained model (model.joblib must exist in the Space root)
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)

# 2. Prepare logging (logs are saved locally and periodically pushed to the Hub)
log_file = Path("logs") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent
log_folder.mkdir(parents=True, exist_ok=True)

# ⚠️ Replace this with your real dataset repo ID on the Hub, e.g.:
# "dariuscyrus7819/insurance_charge_logs"
scheduler = CommitScheduler(
    repo_id="dariuscyrus7819/insurance_charge_logs",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2,  # commit every 2 calls
)

# 3. Define prediction function
def predict(age, sex, bmi, children, smoker, region):
    # Build one-row DataFrame with the same columns used in training
    data = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }
        ]
    )

    # Make prediction
    prediction = model.predict(data)

    # Log inputs and prediction
    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "age": age,
                        "sex": sex,
                        "bmi": bmi,
                        "children": children,
                        "smoker": smoker,
                        "region": region,
                        "prediction": float(prediction[0]),
                    }
                )
            )
            f.write("\n")

    return float(prediction[0])


# 4. Gradio UI components
age_input = gr.Slider(minimum=18, maximum=100, step=1, label="Age")
sex_input = gr.Radio(choices=["male", "female"], label="Sex")
bmi_input = gr.Slider(minimum=10, maximum=60, step=0.1, label="BMI")
children_input = gr.Slider(minimum=0, maximum=5, step=1, label="Number of Children")
smoker_input = gr.Radio(choices=["yes", "no"], label="Smoker")
region_input = gr.Radio(
    choices=["northeast", "northwest", "southeast", "southwest"], label="Region"
)

output = gr.Number(label="Predicted Insurance Charge (USD)")

# 5. Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=[age_input, sex_input, bmi_input, children_input, smoker_input, region_input],
    outputs=output,
    title="HealthyLife Insurance Charge Prediction",
    description="Estimate insurance charges for a prospective HealthyLife customer based on their profile.",
)

# 6. Launch
demo.queue()
demo.launch()
