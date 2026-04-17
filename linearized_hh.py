# Import required modules
import os
import numpy as np
from scipy.integrate import odeint
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

# Values from https://doi.org/10.1016/S0006-3495(69)86446-5

# Constants
C_m  = 1.0 # membrane capacitance, in uF/cm^2
g_Na = 0.0106 #120.0 # maximum conducances, in mS/cm^2; S = mho
g_K  = 0.3667 #36.0
g_L  = 0.3
E_Na = 115.0 #50.0 # Nernst reversal potentials, in mV
E_K  = -12.0 #-77.0
E_L  = 10.5989 #-54.387

# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(V):
	x = (V + 40.0) / 10.0
	return np.where(np.abs(x) < 1e-7, 1.0, 0.1 * (V + 40.0) / -np.expm1(-x))
def beta_m(V):
	return 4.0*np.exp(-(V+65.0) / 18.0)
def alpha_h(V):
	return 0.07*np.exp(-(V+65.0) / 20.0)
def beta_h(V):
	return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
def alpha_n(V):
	x = (V + 55.0) / 10.0
	return np.where(np.abs(x) < 1e-7, 0.1, 0.01 * (V + 55.0) / -np.expm1(-x))
def beta_n(V):
	return 0.125*np.exp(-(V+65) / 80.0)

# Linearized membrane currents
def deltaI_K(V, deltaV, deltan):
	return G_K(V) * deltaV + (4 * g_K * (n_inf(V)**3) * (V - E_K) * deltan)
def deltaI_Na(V, deltaV, deltam, deltah):
	return G_Na(V) * deltaV + (3 * g_Na * (m_inf(V)**2) * h_inf(V) * (V - E_Na) * deltam) + (g_Na * (m_inf(V)**3) * (V - E_Na) * deltah)

# External current
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

# Linearized elements of the ionic channels
def g_n(V):
	return ((4 * g_K * (n_inf(V)**3) * (V - E_K)) * (dalphan_dV(V) - n_inf(V) * (dalphan_dV(V) + dbetan_dV(V)))) / (alpha_n(V) + beta_n(V))
def L_n(V):
	return 1 / (g_n(V) * (alpha_n(V) + beta_n(V)))
def g_m(V):
	return (3 * g_Na * (m_inf(V)**2) * h_inf(V) * (V - E_Na)) * (dalpham_dV(V) - m_inf(V) * (dalpham_dV(V) + dbetam_dV(V))) / (alpha_m(V) + beta_m(V))
def L_m(V):
	return 1 / (g_m(V) * (alpha_m(V) + beta_m(V)))
def g_h(V):
	return (g_Na * (m_inf(V)**3) * (V - E_Na)) * (dalphah_dV(V) - h_inf(V) * (dalphah_dV(V) + dbetah_dV(V))) / (alpha_h(V) + beta_h(V))
def L_h(V):
	return 1 / (g_h(V) * (alpha_h(V) + beta_h(V)))

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

def Linear_dALLdt(X, t):
	V = 0
	deltaV, deltam, deltah, deltan = X

	# Calculate membrane potential and gating variables
	ddeltaVdt = (I_inj(t) - (g_L * deltaV) - deltaI_K(V, deltaV, deltan) - deltaI_Na(V, deltaV, deltam, deltah)) / C_m
	ddeltandt = dalphan_dV(V) * deltaV - (alpha_n(V) + beta_n(V)) * deltan - n_inf(V) * dalphan_dV(V) * deltaV - n_inf(V) * dbetan_dV(V) * deltaV
	ddeltamdt = dalpham_dV(V) * deltaV - (alpha_m(V) + beta_m(V)) * deltam - m_inf(V) * dalpham_dV(V) * deltaV - m_inf(V) * dbetam_dV(V) * deltaV
	ddeltahdt = dalphah_dV(V) * deltaV - (alpha_h(V) + beta_h(V)) * deltah - h_inf(V) * dalphah_dV(V) * deltaV - h_inf(V) * dbetah_dV(V) * deltaV

	return ddeltaVdt, ddeltamdt, ddeltahdt, ddeltandt

if __name__ == '__main__':
	# The time to integrate over
	t = np.arange(0.0, 400.0, 0.1)

	# Initial conditions: zero perturbation for deltaV; steady-state gating at V=0
	X = odeint(Linear_dALLdt, [0, 0.0529, 0.5961, 0.3177], t)

	V = X[:,0]
	m = X[:,1]
	h = X[:,2]
	n = X[:,3]

	plt.figure(figsize=(8, 8))

	plt.subplot(3,1,1)
	plt.title('Hodgkin-Huxley Neuron')
	plt.plot(t, V)
	plt.ylabel('V (mV)')
	plt.grid(which='both', axis='both', linestyle=':')

	plt.subplot(3,1,2)
	plt.plot(t, m, label='m')
	plt.plot(t, h, label='h')
	plt.plot(t, n, label='n')
	plt.ylabel('Gating Value')
	plt.grid(which='both', axis='both', linestyle=':')
	plt.legend()

	plt.subplot(3,1,3)
	plt.plot(t, I_inj(t))
	plt.xlabel('Time (ms)')
	plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
	plt.ylim(-1, 31)
	plt.grid(which='both', axis='both', linestyle=':')

	plt.tight_layout()
	plt.savefig('linearized_hh.pdf', format='pdf', dpi=150)
	plt.close()