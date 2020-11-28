# %%
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

# %%
data = pd.read_csv('d.csv')
# %%
display(data)
display(data.describe())

fig, ax = plt.subplots()
ax.set(xlabel='x', ylabel='y')
ax.scatter(data['x'], data['y'])

# %%
with pm.Model() as model:
    beta1 = pm.Normal("beta1", mu=0, sigma=100)
    beta2 = pm.Normal("beta2", mu=0, sigma=100)
    _lambda = beta1 + beta2 * data['x'].values
    obs = pm.Poisson("y", mu=np.exp(_lambda), observed=data['y'].values)

    # draws: サンプリング回数
    # tune: チューニングのためのサンプリング回数
    # chains: サンプル列の数
    # core: 計算に使うコア数
    # return_inferencedata: arviz.InferenceDataで返すかどうか
    idata = pm.sample(
        2000,
        tune=1000,
        chains=3,
        cores=1,
        random_seed=15,
        return_inferencedata=True
    )

# %%
# MCMCの結果や過程をプロット
az.plot_trace(idata)
az.summary(idata)
# %%
plt.scatter(data['x'], data['y'])
# %%
