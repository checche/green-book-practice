# %%
import arviz as az
from IPython.core.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import binom

# %%
data = pd.read_csv('data7a.csv')

# %%
display(data)
display(data.describe())
hist = np.histogram(data['y'], range=(0, 9), bins=9)
display(hist)

# %%
# 全然2項分布になっていない。直感的には4個が多いはず
# 過分散がある。
fig, ax = plt.subplots()
ax.set(xlabel='y', ylabel='count')
ax.scatter(range(9), hist[0])

# %%
with pm.Model() as model:
    beta = pm.Normal('beta', mu=0, sigma=100)
    s = pm.Uniform('s', lower=0, upper=10000)
    r = pm.Normal('r', mu=0, sigma=s, shape=len(data))
    q = pm.math.sigmoid(beta + r)
    y = pm.Binomial('y', n=8, p=q, observed=data['y'])

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
# az.plot_trace(idata)
az.summary(idata)

# %%
# サンプル列の表示
display(idata.posterior["beta"])
display(idata.posterior["s"])

# %%
# 推定されたパラメータの分布やr_hatの表示
az.plot_forest(idata, r_hat=True)
az.plot_posterior(idata)

# %%
beta_means = idata.posterior['beta'].values.mean(axis=1)
beta_means

# %%
s_means = idata.posterior['s'].values.mean(axis=1)
s_means

# %%
r_means = idata.posterior['r'].values.mean(axis=1)
r_means

# %%
# 3つのsごとにrを生成 3x100
r_sims = np.array([np.random.normal(loc=0, scale=s_mean, size=100)
                   for s_mean in s_means])
r_sims

# %%
# それぞれのrからqを求める 3x100
z_sims = beta_means.reshape(3, 1) + r_sims
q_sims = 1 / (1 + np.exp(-z_sims))
nums = np.arange(9)

# %%
# 3本のchainそれぞれの、100個体の生起確率qにおいて
# 0-8のなかでiになる確率を求める
# (chain, y, sample) = (3, 9, 100)
count_tmp = np.array([
    [
        [binom.pmf(i, 8, q_tmp) for q_tmp in q_sim]
        for i in nums]
    for q_sim in q_sims]
)

# %%
# 各個体の種子数がiになる確率
# (y, sample) = (9, 100)
count_sims = count_tmp.mean(axis=0)
count_sims.shape
count_sims

# %%
# 個体間の平均を取ることでiごとの平均的な存在確率(つまり推測された確率分布)がわかる
# (sample) = (9)
count_sim = count_sims.mean(axis=1)*100  # もとのデータ数が100なので100倍している
count_sim

# %%
fig, ax = plt.subplots()
ax.set(xlabel='x', ylabel='y')
ax.plot(range(9), hist[0], marker='o', label='data')
ax.plot(range(9), count_sim, marker='x', label='predicted')
ax.legend()
