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
