import pandas as pd


calm = pd.read_csv("../simulated_data/calm.csv")

agitated = pd.read_csv("../simulated_data/agitated.csv")

depressed = pd.read_csv("../simulated_data/depressed.csv")

calm["label"] = "calm"
agitated["label"] = "agitated"
depressed["label"] = "depressed"

dataset = pd.concat([calm, agitated, depressed], ignore_index=True)
print(dataset.head())
dataset.to_csv("physio_labeled.csv", index=False)
df = pd.read_csv("physio_labeled.csv")
