# Import required modules
import os
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Constants
C_m  = 1.0 # membrane capacitance, in uF/cm^2
g_Na = 120.0 # maximum conducances, in mS/cm^2
g_K  = 36.0
g_L  = 0.3
E_Na = 50.0 # Nernst reversal potentials, in mV
E_K  = -77.0
E_L  = -54.387

# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(V):
	x = (V + 40.0) / 10.0
	return np.where(np.abs(x) < 1e-7, 1.0, 0.1 * (V + 40.0) / -np.expm1(-x))
def beta_m(V):
	return 4.0*sp.exp(-(V+65.0) / 18.0)
def alpha_h(V):
	return 0.07*sp.exp(-(V+65.0) / 20.0)
def beta_h(V):
	return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
def alpha_n(V):
	x = (V + 55.0) / 10.0
	return np.where(np.abs(x) < 1e-7, 0.1, 0.01 * (V + 55.0) / -np.expm1(-x))
def beta_n(V):
	return 0.125*sp.exp(-(V+65) / 80.0)

# Membrane currents (in uA/cm^2)
#  Sodium (Na = element name)
def I_Na(V, m, h):
	return g_Na * m**3 * h * (V - E_Na) #* 1.9
#  Potassium (K = element name)
def I_K(V, n):
	return g_K * n**4 * (V - E_K) #* 1.9
#  Leak
def I_L(V):
	return g_L * (V - E_L)

def I_inj(t):
	return 10*(t>100) - 10*(t>300)

# Steady-state gating variables
def n_inf(V):
	return alpha_n(V) / (alpha_n(V) + beta_n(V))
def m_inf(V):
	return alpha_m(V) / (alpha_m(V) + beta_m(V))
def h_inf(V):
	return alpha_h(V) / (alpha_h(V) + beta_h(V))

# Pure conductances
def G_K(V):
	return g_K * (n_inf(V) ** 4)
def G_Na(V):
	return g_Na * (m_inf(V) ** 3) * h_inf(V)

# The time to integrate over
t = sp.arange(0.0, 400.0, 0.1)

# Integrate
def dALLdt(X, t):
	V, m, h, n = X
	
	# Calculate membrane potential and gating variables
	dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
	dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
	dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
	dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
	return dVdt, dmdt, dhdt, dndt

# Initial conditions: V_rest = -65 mV, m_inf(-65), h_inf(-65), n_inf(-65)
X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)

# Split results into their respective variables
V = X[:,0]
m = X[:,1]
h = X[:,2]
n = X[:,3]
ina = I_Na(V, m, h)
ik = I_K(V, n)
il = I_L(V)

# Plotting parameters
plt.figure(figsize=(8, 10))

plt.subplot(4,1,1)
plt.title('Hodgkin-Huxley Neuron')
plt.plot(t, V)
plt.ylabel('V (mV)')
plt.grid(which='both', axis='both', linestyle=':')

plt.subplot(4,1,2)
plt.plot(t, ina, label='$I_{Na}$')
plt.plot(t, ik, label='$I_{K}$')
plt.plot(t, il, label='$I_{L}$')
plt.ylabel('Current')
plt.grid(which='both', axis='both', linestyle=':')
plt.legend()

plt.subplot(4,1,3)
plt.plot(t, m, label='m')
plt.plot(t, h, label='h')
plt.plot(t, n, label='n')
plt.ylabel('Gating Value')
plt.grid(which='both', axis='both', linestyle=':')
plt.legend()

plt.subplot(4,1,4)
plt.plot(t, I_inj(t))
plt.xlabel('Time (ms)')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, 31)
plt.grid(which='both', axis='both', linestyle=':')

plt.tight_layout()
# plt.show()
plt.savefig('original_hh.pdf', format='pdf', dpi=150)
plt.close()