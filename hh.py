# Import required modules
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy import stats
import scipy.linalg as lin
import matplotlib       # To use plt.show(), comment this line.
matplotlib.use('Agg')   # To use plt.show(), comment this line.
import matplotlib.pyplot as plt

# Plotting settings
plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 12})

# Constants
C_m  =   1.0 # membrane capacitance, in uF/cm^2
g_Na = 120.0 # maximum conducances, in mS/cm^2
g_K  =  36.0
g_L  =   0.3
E_Na =  50.0 # Nernst reversal potentials, in mV
E_K  = -77.0
E_L  = -54.387

# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(V):
	return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))
def beta_m(V):
	return 4.0*sp.exp(-(V+65.0) / 18.0)
def alpha_h(V):
	return 0.07*sp.exp(-(V+65.0) / 20.0)
def beta_h(V):
	return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))
def alpha_n(V):
	return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))
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

# Derivatives of the channel gating kinetics
def dalphan_dV(V):
	return (0.01 / (1 - np.exp(-(V/10)-(11/2)))) - (((0.01 * V + 0.55) * np.exp(-(V/10)-(11/12))) / (10 * ((1 - np.exp(-(V/10)-(11/2))) ** 2)))
def dbetan_dV(V):
	return -0.0015625 * np.exp(-(V/80)-(13/16))
def dalpham_dV(V):
	return (((0.1 * V + 4.0) * np.exp(-(V/10) - 4)) / (10 * ((np.exp(-(V/10) - 4) + 1) ** 2))) + (0.1 / (np.exp(-(V/10) - 4) + 1))
def dbetam_dV(V):
	return -np.exp(-(V/20) - (13/4)) / 5
def dalphah_dV(V):
	return -0.0035 * np.exp(-(V/20) - (13/4))
def dbetah_dV(V):
	return np.exp(-(V/20)-(7/4)) / (20 * ((np.exp(-(V/20)-(7/4)) + 1) ** 2))

# Coil-like elements for linearization
def g_n(V, n):
	return ((4 * g_K * n**3 * (V - E_K)) * (dalphan_dV(V) - n * (dalphan_dV(V) + dbetan_dV(V)))) / (alpha_n(V) + beta_n(V))
def L_n(V, n):
	return 1 / (g_n(V, n) * (alpha_n(V) + beta_n(V)))
def g_m(V, m, h):
	return (3 * g_Na * (m**2) * h * (V - E_Na)) * (dalpham_dV(V) - m * (dalpham_dV(V) + dbetam_dV(V))) / (alpha_m(V) + beta_m(V))
def L_m(V, m, h):
	return 1 / (g_m(V, m, h) * (alpha_m(V) + beta_m(V)))
def g_h(V, m, h):
	return (g_Na * (m**3) * (V - E_Na)) * (dalphah_dV(V) - h * (dalphah_dV(V) + dbetah_dV(V))) / (alpha_h(V) + beta_h(V))
def L_h(V, m, h):
	return 1 / (g_h(V, m, h) * (alpha_h(V), beta_h(V)))

# Steady-state gating variables
def n_inf(V):
	return alpha_n / (alpha_n(V) + beta_n(V))
def m_inf(V):
	return alpha_m / (alpha_m(V) + beta_m(V))
def h_inf(V):
	return alpha_h / (alpha_h(V) + beta_h(V))

# Pure conductances
def G_K(V):
	return g_K * (n_inf(V) ** 4) * V
def G_Na(V, m, h):
	return g_Na * (m ** 3) * h

# Current on a RL branch
def current_RL(V, g, L, T):
	R = 1/g
	tau = R * L
	return V * (1 - np.exp(-T/tau)) / R

# The time to integrate over
t = sp.arange(0.0, 400.0, 0.1)

# Integrate
def Linear_dALLdt(X, t):
	V, m, h, n = X

	# Calculate membrance potential and gating variables
	dVdt = (I_inj(t) - I_L(V) - (G_K(V) * V + current_RL(V, g_n(V, n), L_n(V, n), t)) - (G_Na(V, m, h) * V + current_RL(V, g_m(V, m, h), L_m(V, m, h), t) + current_RL(V, g_h(V, m, h), L_h(V, m, h), t)))
	dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
	dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
	dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
	return dVdt, dmdt, dhdt, dndt

# Integrate
def dALLdt(X, t):
	V, m, h, n = X
	
	# Calculate membrane potential and gating variables
	dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
	dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
	dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
	dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
	return dVdt, dmdt, dhdt, dndt

X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
X = odeint(Linear_dALLdt, [-65, 0.05, 0.6, 0.32], t)

# Split results into their respective variables
V = X[:,0]
m = X[:,1]
h = X[:,2]
n = X[:,3]
ina = I_Na(V, m, h)
ik = I_K(V, n)
il = I_L(V)

# Plotting parameters
plt.figure()

plt.subplot(4,1,1)
plt.title('Hodgkin-Huxley Neuron')
plt.plot(t, V, 'k')
plt.ylabel('V (mV)')
plt.grid(which='both', axis='both', linestyle=':')

plt.subplot(4,1,2)
plt.plot(t, ina, 'c', label='$I_{Na}$')
plt.plot(t, ik, 'y', label='$I_{K}$')
plt.plot(t, il, 'm', label='$I_{L}$')
plt.ylabel('Current')
plt.grid(which='both', axis='both', linestyle=':')
plt.legend()

plt.subplot(4,1,3)
plt.plot(t, m, 'r', label='m')
plt.plot(t, h, 'g', label='h')
plt.plot(t, n, 'b', label='n')
plt.ylabel('Gating Value')
plt.grid(which='both', axis='both', linestyle=':')
plt.legend()

plt.subplot(4,1,4)
plt.plot(t, I_inj(t), 'k')
plt.xlabel('t (ms)')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
plt.ylim(-1, 31)
plt.grid(which='both', axis='both', linestyle=':')

plt.tight_layout()
# plt.show()
plt.savefig('hh.pdf', format='pdf', dpi=150)
plt.close()