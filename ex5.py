import numpy as np
import scipy.stats as stats

N1, N2, N3 = 350, 340, 310
alt_obs = np.array([N1, N2, N3])
N_options = len(alt_obs)
N_tot = sum(alt_obs)
alt_probs = alt_obs/N_tot
null_probs = np.ones(N_options)/N_options
null_obs = null_probs*N_tot

alt_likelihood = stats.multinomial.pmf(alt_obs, N_tot, alt_probs)
null_likelihood = stats.multinomial.pmf(alt_obs, N_tot, null_probs)
likelihood_ratio = alt_likelihood / null_likelihood
print(f'likelihood ratio: {likelihood_ratio}')

alpha = 0.05
critical_value = stats.chi2.ppf(1 - alpha, df=N_options - 1)
loglike_ratio = 2*np.log(likelihood_ratio)
print(f'likelihood test, reject null hypothesis?: {loglike_ratio>critical_value}')

chi_square = stats.chisquare(alt_obs)[0]
chi_square = sum((alt_obs - null_obs) ** 2 / null_obs)
p_value = stats.chi2.sf(chi_square, df=N_options - 1)
print(f'chi_square test, reject null hypothesis?: {chi_square>critical_value}')
print(f'p_value test, reject null hypothesis?: {p_value<alpha}')
