# Dexter Dysthe
# Dr. Johannes
# B9337
# 15 November 2021

# Square Root SV: dS_t = (r_t + mu_t) * S_t dt + sqrt(V_t) * S_t dW_t,S
#                 dV_t = kappa_v * (theta_v - V_t) dt + sigma_v sqrt(V_t) dW_t,V
#
# Even though Das and Sundaram do not restrict the model to the case dW_t,S dW_t,V = 0,
# i.e. W_t,S and W_t,V are uncorrelated, given we have done so far in lecture I will
# implement the model in the case of the Brownian motions being independent.
#
# We consider the parameter values as in Table 7 of the paper. That is, we consider:
#               (i)   S_0 = 100
#               (ii)  V_0 = 0.0125
#               (iii) kappa_v = 1, 5
#               (iv)  sigma_v = 0.1, 0.4
#               (v)   theta_v = 0.01
#               (vi)  r_t = 0
#               ** I could not find values for mu_t in the paper, and so I decided on 6% as in HW2
#               (vii) mu_t = 0.06

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set()
sns.set_style('dark')
sns.set_style('ticks')
sns.despine()


def sv_monte_carlo(nsim, kappa_v, sigma_v):
    all_sv_paths = []
    for y in range(nsim):
        sv = [0.0125]

        for x in range(999):
            step_size = 1/1000
            dW = np.random.normal(0, np.sqrt(step_size))
            sv_drift = kappa_v * (0.01 - sv[x]) * step_size
            sv_diffusion = sigma_v * np.sqrt(sv[x]) * dW

            if sv[x] + sv_drift + sv_diffusion < 0:
                sv.append(0)
            else:
                sv.append(sv[x] + sv_drift + sv_diffusion)

        if nsim > 1:
            all_sv_paths.append(sv)
        elif nsim == 1:
            return sv

    return np.array(all_sv_paths).transpose()


def stock_returns_monte_carlo(nsim, kappa_v, sigma_v):
    all_return_paths = []
    for y in range(nsim):
        stock = [100]
        sv_path = sv_monte_carlo(1, kappa_v, sigma_v)

        for x in range(1000):
            step_size = 1/1000
            dW = np.random.normal(0, np.sqrt(step_size))
            stock_drift = 0.06 * stock[x] * step_size
            stock_diffusion = np.sqrt(sv_path[x]) * stock[x] * dW

            stock.append(stock[x] + stock_drift + stock_diffusion)

        stock_returns = [stock[i+1] / stock[i] for i in range(1000)]
        all_return_paths.append(stock_returns)

    return np.array(all_return_paths).transpose()


# Creating a list of sample points for time axis over the unit interval
t = [k/1000 for k in range(1000)]


print('\n')
# ------------------------ Simulating Variance ---------------------------- #

# Number of sims = 10
sv_sample_paths1 = sv_monte_carlo(50, 1, 0.1)
plt.plot(t, sv_sample_paths1)
plt.title('Sample Paths of Square Root Variance Process (kappa_v = 1, sigma_v = 0.1)')
plt.xlabel('Time')
plt.show()

sv_sample_paths2 = sv_monte_carlo(50, 1, 0.4)
plt.plot(t, sv_sample_paths2)
plt.title('Sample Paths of Square Root Variance Process (kappa_v = 1, sigma_v = 0.4)')
plt.xlabel('Time')
plt.show()

sv_sample_paths3 = sv_monte_carlo(50, 5, 0.1)
plt.plot(t, sv_sample_paths3)
plt.title('Sample Paths of Square Root Variance Process (kappa_v = 5, sigma_v = 0.1)')
plt.xlabel('Time')
plt.show()

sv_sample_paths4 = sv_monte_carlo(50, 5, 0.4)
plt.plot(t, sv_sample_paths4)
plt.title('Sample Paths of Square Root Variance Process (kappa_v = 5, sigma_v = 0.4)')
plt.xlabel('Time')
plt.show()


print('\n')
# ------------------------ Simulating Stock Returns ---------------------------- #

# Number of sims = 10
stock_returns_sample_paths1 = stock_returns_monte_carlo(1, 1, 0.1)
plt.plot(t, stock_returns_sample_paths1)
plt.title('Sample Paths of Stock Returns Process (kappa_v = 1, sigma_v = 0.1)')
plt.xlabel('Time')
plt.show()

stock_returns_sample_paths2 = stock_returns_monte_carlo(1, 1, 0.4)
plt.plot(t, stock_returns_sample_paths2)
plt.title('Sample Paths of Stock Returns Process (kappa_v = 1, sigma_v = 0.4)')
plt.xlabel('Time')
plt.show()

stock_returns_sample_paths3 = stock_returns_monte_carlo(1, 5, 0.1)
plt.plot(t, stock_returns_sample_paths3)
plt.title('Sample Paths of Stock Returns Process (kappa_v = 5, sigma_v = 0.1)')
plt.xlabel('Time')
plt.show()

stock_returns_sample_paths4 = stock_returns_monte_carlo(1, 5, 0.4)
plt.plot(t, stock_returns_sample_paths4)
plt.title('Sample Paths of Stock Returns Process (kappa_v = 5, sigma_v = 0.4)')
plt.xlabel('Time')
plt.show()




