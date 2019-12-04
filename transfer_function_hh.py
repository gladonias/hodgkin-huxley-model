from scipy import signal, constants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plotting settings
plt.style.use('seaborn-deep')
plt.rcParams.update({'font.size': 12})

# Declare variables
R_l = 3.33e3
R_K = 2.72e3
r_n = 1.2e3
L_n = 6.43
r_h = 1.4e4
L_h = 1.19e2
C = 1e6
R_Na = 5.313e5
L_m = -230
r_m = -2.3e3

G_T = 1/R_l + 1/R_K + 1/R_Na
g_n = 1/r_n
g_m = 1/r_m
g_h = 1/r_h
lamb = L_n + L_m + L_h
gamma = G_T + g_n + g_m + g_h

num = [gamma/C, 0]
den = [gamma, (gamma**2)/C, gamma/(lamb*C)]

sys = signal.TransferFunction(num, den)
w, mag, phase = signal.bode(sys)

hz = w / (2 * constants.pi)

plt.figure()
plt.semilogx(hz, mag)    # Bode magnitude plot
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid(which='both', axis='both', linestyle=':')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig('mag_bode.pdf', format='pdf', dpi=150)
plt.close()

plt.figure()
plt.semilogx(hz, phase)  # Bode phase plot
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [deg]')
plt.grid(which='both', axis='both', linestyle=':')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig('phase_bode.pdf', format='pdf', dpi=150)
plt.close()