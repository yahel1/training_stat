import numpy as np
import scipy.stats as stats

lambda0 = 1
n_obs = 2000
alpha = 0.05
critical_value = stats.norm.ppf(1 - alpha / 2)
n_repeat = 10000
test_results = np.zeros(n_repeat)
for ind in range(n_repeat):
    obs = np.random.poisson(lambda0, n_obs)
    mean_obs = np.mean(obs)
    se_obs = stats.sem(obs)
    W = np.abs((lambda0 - mean_obs) / se_obs)
    is_H1 = W > critical_value
    test_results[ind] = is_H1
alpha_calc = np.sum(test_results == 1)/n_repeat
print(f'calculated type 1 error: {alpha_calc}')
