import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

Tf, Ts, Tus = 1, 50, 2500
alpha = [-2, 2, -1.5, 1.5]
delta = [0, 0, -1.5, -1.5]
Iapp = -1.6
g_inh = -0.3
b, dsyn = 5, -1
tmax = 20000
ti = 15000
spike_thresh = -2
servo_scale = 30.0

def sigmoid(x):
    return 1/(1+np.exp(-b*(x-dsyn)))

def neuron_eqs(S, I):
    vm, vf, vs, vus = S
    dvm = -vm - alpha[0]*np.tanh(vf-delta[0]) \
              - alpha[1]*np.tanh(vs-delta[1]) \
              - alpha[2]*np.tanh(vs-delta[2]) \
              - alpha[3]*np.tanh(vus-delta[3]) + I
    dvf = (vm - vf)/Tf
    dvs = (vm - vs)/Ts
    dvus = (vm - vus)/Tus
    return [dvm, dvf, dvs, dvus]

def make_network(N):
    asyn = g_inh * np.ones((N, N))
    np.fill_diagonal(asyn, 0)

    def network(t, S):
        dS = []
        Vs = [S[i*4+2] for i in range(N)]   # use "vs" for synaptic input
        Isyn = asyn @ sigmoid(np.array(Vs))
        for i in range(N):
            Si = S[i*4:(i+1)*4]
            dS.extend(neuron_eqs(Si, Iapp+Isyn[i]))
        return dS
    return network

fig, axes = plt.subplots(2, 4, figsize=(18, 6), sharex='col')

for idx, N in enumerate([3, 4, 5, 6]):
    S0 = np.zeros(N*4)
    for i in range(N):
        S0[i*4] = -1 + 0.1*np.random.randn()

    t_eval = np.linspace(0, tmax, tmax)
    sol = solve_ivp(make_network(N), (0, tmax), S0,
                    method="BDF", t_eval=t_eval)

    for i in range(N):
        Vm = sol.y[i*4, :]
        axes[0, idx].plot(sol.t[ti:], Vm[ti:] + 6*(N-i-1), lw=0.8)

    axes[0, idx].set_title(f"N={N}")
    if idx == 0:
        axes[0, idx].set_ylabel("Vm (shifted)")


    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    decoded = []
    prev_vm = sol.y[0::4, 0].copy()

    print(f"\n--- Decoded angles for N={N} ---")
    for k, t in enumerate(sol.t):
        spk = []
        for i in range(N):
            vm = sol.y[i*4, k]
            if vm > spike_thresh and prev_vm[i] <= spike_thresh:
                spk.append(1)
            else:
                spk.append(0)
            prev_vm[i] = vm

        spk = np.array(spk)
        if np.sum(spk) > 0:
            x = np.sum(spk * np.cos(phases))
            y = np.sum(spk * np.sin(phases))
            theta = np.degrees(np.arctan2(y, x)) % 360
        else:
            theta = decoded[-1] if decoded else 0
        decoded.append(theta)

        if k % 100 == 0:
            servo_angle = servo_scale * np.sin(np.radians(theta))
            print(f"t={int(t):5d} ms | raw angle={theta:6.1f}° | servo={servo_angle:6.2f}°")
    

    axes[1, idx].plot(sol.t[ti:], decoded[ti:], 'r')
    if idx == 0:
        axes[1, idx].set_ylabel("Decoded Angle [deg]")
    axes[1, idx].set_xlabel("Time [a.u.]")

plt.tight_layout()
plt.show()

