from data_simulation import simulate_physio
from plot import plot_physio_signals
import pandas as pd

calm = pd.read_csv("../simulated_data/calm.csv")
plot_physio_signals(calm, "Calm")

agitated = pd.read_csv("../simulated_data/agitated.csv")
plot_physio_signals(agitated, "Agitated")

depressed = pd.read_csv("../simulated_data/depressed.csv")
plot_physio_signals(depressed, "Depressed")
