import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st

samples = 20
x = np.linspace(0, 10, samples)

##Prior
mean_0 = 4
sd_0 = 0.8
prior_dist = st.norm(mean_0, sd_0).pdf(x)

##Sample
mean_x = 6
sd_x = 1.5
sample_dist = dist = st.norm(mean_x, sd_x).pdf(x)

##Posterior
x_t = st.norm(mean_x,sd_x).rvs(samples)
var_n = 1/((1/sd_0**2)+(samples/sd_x**2))
mean_n = var_n * ((mean_0/sd_0**2)+(np.mean(x_t)*samples/sd_x**2))
print("Mean of the posterior distibution is",mean_n,"and variance is",var_n)
posterior_dist = st.norm(mean_n, np.sqrt(var_n)).pdf(x)

##Plot
plt.plot(x,prior_dist,"b-",label='Prior')
plt.plot(x,sample_dist,"g-",label='Sample')
plt.plot(x,posterior_dist,"r-",label='Posterior')
plt.legend(loc='upper left')
plt.title('Probability Density Plot')
plt.ylabel('Probability Density')
plt.xlabel('X')
plt.show()

