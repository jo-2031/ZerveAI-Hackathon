import pandas as pd
import numpy as np

df = pd.read_csv("training_data.csv", low_memory=False)
df = df.replace("", np.nan)
df.info()