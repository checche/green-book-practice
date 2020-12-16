# %%
import arviz as az
from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymc3 as pm
from pymc3.distributions import continuous
from pymc3.distributions import distribution
import theano.tensor as tt


floatX = "float32"
# %%
data = pd.read_csv('Y.csv')
N = len(data)

# %%
display(
    data,
    data.describe()
)
# %%
# 個体差と植木鉢差がある。
display(
    px.scatter(data, y='x')
)
# %%
# preprocess
data['is_raw'] = True

data['adj'] = np.array([
    [1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [10, 12], [11, 13], [12, 14], [13, 15], [14, 16],
    [15, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22],
    [21, 23], [22, 24], [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32], [31, 33], [32, 34],
    [33, 35], [34, 36], [35, 37], [36, 38], [37, 39], [38, 40],
    [39, 41], [40, 42], [41, 43], [42, 44], [43, 45], [44, 46],
    [45, 47], [46, 48], [47, 49], [48]],
    dtype=object
)
data['weight'] = np.array([
    [1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],
    [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0]
], dtype=object)

# %%
wmat = np.zeros((N, N))
amat = np.zeros((N, N))
for i, a in enumerate(data['adj']):
    amat[i, a] = 1
    wmat[i, a] = data['weight'][i]

display(wmat, amat)

# %%


class CAR(distribution.Continuous):
    """
    Conditional Autoregressive (CAR) distribution

    Parameters
    ----------
    a : adjacency matrix
    w : weight matrix
    tau : precision at each location
    """

    def __init__(self, w, a, tau, *args, **kwargs):
        super(CAR, self).__init__(*args, **kwargs)
        self.a = a = tt.as_tensor_variable(a)
        self.w = w = tt.as_tensor_variable(w)
        self.tau = tau*tt.sum(w, axis=1)
        self.mode = 0.

    def logp(self, x):
        tau = self.tau
        w = self.w
        a = self.a

        mu_w = tt.sum(x*a, axis=1)/tt.sum(w, axis=1)
        return tt.sum(continuous.Normal.dist(mu=mu_w, tau=tau).logp(x))


# %%
# なかなか収束しない
with pm.Model() as model:
    beta = pm.Normal('beta', mu=0, sigma=100)

    s = pm.Uniform('s', lower=0, upper=10000)
    tau = 1 / (s * s)

    r = CAR('r', w=wmat, a=amat, tau=tau, shape=N)

    mu = np.exp(beta + r)

    y = pm.Poisson('y', mu=mu, observed=data['x'].values)

    idata = pm.sample(
        2000,
        tune=2000,
        chains=3,
        cores=1,
        random_seed=15,
        return_inferencedata=True,
    )

# %%
pm.model_to_graphviz(model)

# %%
# MCMCの結果や過程をプロット
# az.plot_trace(idata)
az.summary(idata)

# %%
# サンプル列の表示
display(
    idata.posterior["beta"],
    idata.posterior["s"],
)

# %%
# 推定されたパラメータの分布やr_hatの表示
az.plot_forest(idata, r_hat=True)
az.plot_posterior(idata)

# %%
beta_means = idata.posterior['beta'].values.mean(axis=1)
s_means = idata.posterior['s'].values.mean(axis=1)
r_means = idata.posterior['r'].values.mean(axis=1)
display(
    beta_means,
    s_means,
    r_means,
)

# %%
# それぞれのrからlambdaを求める 3x100
lambda_means = np.exp(beta_means.reshape(3, 1) + r_means)
lambda_means.shape

# %%
# 各個体の平均値を算出
lambda_mean = lambda_means.mean(axis=0)
lambda_mean

# %%
data_copied = data.copy()
data_copied['is_raw'] = False
data_copied['x'] = lambda_mean
metrics = pd.concat([data, data_copied])
metrics

# %%
display(
    px.scatter(metrics, y='x', facet_row='is_raw'),
)
fig, ax = plt.subplots()
ax.scatter(data.index, data['x'])
ax.plot(data_copied.index, data_copied['x'])

# %%
hdi = az.hdi(idata, hdi_prob=0.80)
# %%
hdi.r.values + hdi.beta.values
# %%
# 収束していないので範囲がとても広い
lambda_hdi = np.exp(hdi.r.values + hdi.beta.values).T
lambda_hdi


# %%
