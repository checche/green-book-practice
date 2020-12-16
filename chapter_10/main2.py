# %%
import arviz as az
from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pymc3 as pm

# %%
data = pd.read_csv('d1.csv')

# %%
display(
    data,
    data.describe()
)
# %%
# 個体差と植木鉢差がある。
display(
    px.scatter(data, x='id', y='y', color='pot'),
    px.box(data, x='pot', y='y', color='f')
)
# %%
# preprocess
codes, uniques = pd.factorize(data['pot'], sort=True)
data['pot_int'] = codes

codes, uniques = pd.factorize(data['f'], sort=True)
data['f_int'] = codes

data['is_raw'] = True
# %%
with pm.Model() as model:
    beta1 = pm.Normal('beta1', mu=0, sigma=100)
    beta2 = pm.Normal('beta2', mu=0, sigma=100)

    s = pm.Uniform('s', lower=0, upper=10000)
    s_p = pm.Uniform('s_p', lower=0, upper=10000)

    r = pm.Normal('r', mu=0, sigma=s, shape=len(data))
    r_p = pm.Normal('r_p', mu=0, sigma=s_p, shape=data['pot_int'].nunique())

    mu = np.exp(beta1 + beta2 *
                data['f_int'].values + r + r_p[data['pot_int']])
    y = pm.Poisson('y', mu=mu, observed=data['y'].values)

    idata = pm.sample(
        2000,
        tune=1000,
        chains=3,
        cores=1,
        random_seed=15,
        return_inferencedata=True
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
    idata.posterior["beta1"],
    idata.posterior["beta2"],
    idata.posterior["s"],
    idata.posterior["s_p"]
)

# %%
# 推定されたパラメータの分布やr_hatの表示
az.plot_forest(idata, r_hat=True)
az.plot_posterior(idata)

# %%
beta1_means = idata.posterior['beta1'].values.mean(axis=1)
beta2_means = idata.posterior['beta2'].values.mean(axis=1)
s_means = idata.posterior['s'].values.mean(axis=1)
s_p_means = idata.posterior['s_p'].values.mean(axis=1)
r_means = idata.posterior['r'].values.mean(axis=1)
r_p_means = idata.posterior['r_p'].values.mean(axis=1)
display(
    beta1_means,
    beta2_means,
    s_means,
    s_p_means,
    r_means,
    r_p_means
)

# %%
# r, r_pの生成
r_sims = np.array([np.random.normal(loc=0, scale=s_mean, size=100)
                   for s_mean in s_means])


r_p_sims = np.array([np.random.normal(loc=0, scale=s_p_mean, size=10)
                     for s_p_mean in s_p_means])
display(
    r_sims,
    r_p_sims
)


# %%
# それぞれのrからlambdaを求める 3x100
lambda_sims = np.exp(
    beta1_means.reshape(3, 1)
    + beta2_means.reshape(3, 1) * data['f_int'].values
    + r_sims
    + r_p_sims[:, data['pot_int']]
)
lambda_sims.shape

# %%
# 各個体の平均値を算出
lambda_sim = lambda_sims.mean(axis=0)
lambda_sim

# %%
data_copied = data.copy()
data_copied['is_raw'] = False
data_copied['y'] = lambda_sim
metrics = pd.concat([data, data_copied])
metrics

# %%
display(
    px.scatter(metrics, x='id', y='y', color='pot', facet_col='is_raw'),
    px.box(metrics, x='pot', y='y', color='f', facet_col='is_raw'),
)
