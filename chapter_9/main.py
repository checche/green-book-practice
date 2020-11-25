# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

# %%
data = pd.read_csv('d.csv')
# %%
display(data)
display(data.describe())
# %%
plt.scatter(data['x'], data['y'])
# %%
