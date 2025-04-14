import matplotlib.pyplot as plt
import pandas as pd

def plot_physio_signals(data, mood_label="Mood"):
    # Plot Heart Rate
    plt.figure(figsize=(10, 4))
    plt.plot(data["heart_rate"], label="HR", color="steelblue")
    plt.title(f"Heart Rate Over Time ({mood_label})")
    plt.xlabel("Time step")
    plt.ylabel("Heart Rate (bpm)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot GSR
    plt.figure(figsize=(10, 4))
    plt.plot(data["gsr"], label="GSR", color="darkorange")
    plt.title(f"GSR Over Time ({mood_label})")
    plt.xlabel("Time step")
    plt.ylabel("Skin Conductance (µS)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

calm = pd.read_csv("../simulated_data/calm.csv")
plot_physio_signals(calm, "Calm")

agitated = pd.read_csv("../simulated_data/agitated.csv")
plot_physio_signals(agitated, "Agitated")

depressed = pd.read_csv("../simulated_data/depressed.csv")
plot_physio_signals(depressed, "Depressed")
