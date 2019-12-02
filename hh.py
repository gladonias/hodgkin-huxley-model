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
