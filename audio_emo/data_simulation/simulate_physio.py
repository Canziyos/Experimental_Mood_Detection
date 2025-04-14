import numpy as np
import pandas as pd
import os

mood_classes = {
    "calm": {
        "hr_range": (60, 75),
        "gsr_range": (2.5, 5.0)
    },

    "agitated": {
        "hr_range": (80, 110),
        "gsr_range": (5.5, 10.0)
    },

    "depressed": {
        "hr_range": (55, 70),
        "gsr_range": (1.0, 3.0)
    }
}
# print(mood_classes.keys())
# print(mood_classes["calm"]["hr_range"])

def generate_data(mood_classes):
    for mood_name, params in mood_classes.items():
        min_hr, max_hr = params["hr_range"]
        min_gsr, max_gsr = params["gsr_range"]

        hr_values = []
        gsr_values = []

        for _ in range(100):
            hr_value = np.random.uniform(min_hr, max_hr) + np.random.normal(0, 2)
            gsr_value = np.random.uniform(min_gsr, max_gsr) + np.random.normal(0, 0.2)
            hr_values.append(hr_value)
            gsr_values.append(gsr_value)

        data = pd.DataFrame({
            "heart_rate": hr_values,
            "gsr": gsr_values
        })

        os.makedirs("../simulated_data", exist_ok=True)
        file_path = f"../simulated_data/{mood_name}.csv"
        data.to_csv(file_path, index=False)
        print(f"Saved {mood_name} data to {file_path}")

generate_data(mood_classes)