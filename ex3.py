import numpy as np
import scipy.stats as stats
from numpy.core._multiarray_umath import ndarray

k = 6
deg_freedom: int = k - 1
options = np.arange(k)
obs_a: ndarray = np.array([8, 4, 11, 9, 11, 17])
obs_b: ndarray = np.array([80, 40, 110, 90, 110, 170])
all_sign_levels = [0.99, 0.95]
ind = 1
for obs in [obs_a, obs_b]:
    print(ind)
    null_obs = np.ones(k)*sum(obs)/k
    null_prob = 1/k*np.ones(k)
    alt_prob = obs / sum(obs)
    chi_square = stats.chisquare(obs)[0]
    chi_square = sum((obs - null_obs) ** 2 / null_obs)
    p_value = stats.chi2.sf(chi_square, df=deg_freedom)
    null_loglikelihood = stats.multinomial.logpmf(obs, sum(obs), null_prob)
    alt_loglikelihood = stats.multinomial.logpmf(obs, sum(obs), alt_prob)
    loglikelihood_chi_squared = 2 * (alt_loglikelihood - null_loglikelihood)
    for sign_level in all_sign_levels:
        alpha = 1 - sign_level
        critical_value = stats.chi2.ppf(sign_level, df=deg_freedom)
        print(f'alpha: {alpha:.2f}, critical_value: {critical_value:.2f}, chi squared: {chi_square:.2f}')
        print('chi square test: is the null hypothesis rejected?', chi_square > critical_value)
        print('likelihood test: is the null hypothesis rejected?', loglikelihood_chi_squared > critical_value)
    print("loglike chi squared:", loglikelihood_chi_squared)
    print(f'p value: {p_value:.2f}')
    ind += 1

