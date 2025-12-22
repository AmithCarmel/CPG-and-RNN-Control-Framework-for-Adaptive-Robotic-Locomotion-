import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import pickle


wkF = np.array([
    [9,49,67,38,24,20,27,14],
    [8,50,68,39,28,21,22,14],
    [10,51,70,41,26,22,18,14],
    [12,52,69,42,24,24,11,14],
    [14,52,63,44,22,26,1,15],
    [16,53,53,45,21,27,-2,15],
    [18,53,41,46,20,29,-1,15],
    [21,54,26,47,18,30,6,16],
    [22,54,23,48,18,32,9,17],
    [25,54,20,49,16,34,12,18],
    [26,54,17,51,16,37,17,18],
    [28,54,14,52,15,39,22,19],
    [30,52,11,54,14,45,27,19],
    [32,54,11,54,14,44,29,20],
    [33,58,13,55,15,36,27,21],
    [34,61,16,56,15,31,24,23],
    [36,64,18,56,14,24,23,24],
    [38,66,20,57,14,20,22,26],
    [39,67,22,57,14,16,21,28],
    [41,64,24,57,14,5,20,30],
    [42,55,26,57,14,-1,19,32],
    [44,44,28,57,15,-3,18,35],
    [45,30,29,57,15,1,18,38],
    [46,21,31,57,15,5,17,40],
    [47,19,32,56,16,9,17,43],
    [48,16,35,57,17,12,16,44],
    [49,12,37,62,18,17,14,35],
    [49,9,39,66,20,24,14,29],
    [50,8,40,68,21,28,14,23],
    [51,10,42,70,22,26,14,19],
    [52,12,43,70,24,24,15,17],
    [52,14,44,67,26,22,15,5],
    [53,16,46,59,27,21,15,-2],
    [53,18,47,47,29,20,16,-2],
    [54,21,48,34,30,18,16,1],
    [54,22,49,24,32,18,17,6],
    [54,25,50,21,34,16,18,10],
    [54,26,51,19,37,16,19,12],
    [54,28,52,15,39,15,20,19],
    [52,30,54,12,45,14,19,24],
    [54,32,55,12,44,14,20,27],
    [58,33,55,11,36,15,22,29],
    [61,34,56,14,31,15,24,26],
    [64,36,56,17,24,14,25,24],
    [66,38,57,18,20,14,27,23],
    [67,39,57,21,16,14,29,21],
    [64,41,57,23,5,14,31,20],
    [55,42,57,24,-1,14,33,20],
    [44,44,57,26,-3,15,36,19],
    [30,45,57,28,1,15,39,18],
    [21,46,56,30,5,15,42,17],
    [19,47,56,32,9,16,45,17],
    [16,48,59,33,12,17,41,17],
    [12,49,64,35,17,18,33,16]
]).reshape(-1,8)

crF = np.array([
    [42,73,83,75,-43,-42,-49,-41],
    [37,75,78,77,-41,-41,-50,-41],
    [36,78,73,80,-38,-41,-50,-41],
    [37,81,68,82,-36,-41,-50,-40],
    [41,83,62,85,-36,-40,-48,-40],
    [42,88,57,85,-36,-40,-47,-39],
    [45,92,51,88,-37,-44,-44,-38],
    [48,91,50,90,-38,-45,-41,-37],
    [51,89,55,91,-39,-48,-40,-36],
    [53,84,55,93,-40,-49,-40,-35],
    [56,80,58,99,-40,-50,-41,-37],
    [59,75,61,101,-41,-50,-41,-40],
    [62,69,64,101,-41,-50,-41,-42],
    [64,64,65,99,-41,-49,-41,-44],
    [67,58,68,95,-42,-48,-42,-46],
    [70,53,71,92,-42,-47,-42,-47],
    [73,48,74,88,-42,-45,-41,-48],
    [74,42,75,83,-41,-43,-41,-49],
    [77,37,78,78,-41,-41,-41,-50],
    [80,36,81,73,-41,-38,-41,-50],
    [82,37,83,68,-40,-36,-40,-50],
    [85,41,85,62,-40,-36,-40,-48],
    [89,42,87,57,-41,-36,-39,-47],
    [93,45,89,51,-45,-37,-38,-44],
    [92,48,91,50,-47,-38,-37,-41],
    [88,51,93,55,-48,-39,-36,-40],
    [83,53,96,55,-49,-40,-35,-40],
    [78,56,100,58,-50,-40,-38,-41],
    [73,59,102,61,-50,-41,-42,-41],
    [68,62,99,64,-50,-41,-43,-41],
    [62,64,96,65,-48,-41,-45,-41],
    [57,67,93,68,-47,-42,-47,-42],
    [51,70,89,71,-46,-42,-48,-42],
    [46,73,84,74,-45,-42,-49,-41]
]).reshape(-1,8)

def sigmoid(x, b=5.0, dsyn=-1.0):
    return 1.0 / (1.0 + np.exp(-b*(x-dsyn)))

def neuron_eqs(S, I, alpha, delta, Tf, Ts, Tus):
    vm, vf, vs, vus = S
    dvm = (-vm - alpha[0]*np.tanh(vf-delta[0]) - alpha[1]*np.tanh(vs-delta[1])
           - alpha[2]*np.tanh(vs-delta[2]) - alpha[3]*np.tanh(vus-delta[3]) + I)
    dvf = (vm - vf)/Tf
    dvs = (vm - vs)/Ts
    dvus = (vm - vus)/Tus
    return [dvm, dvf, dvs, dvus]

def make_network(N, alpha, delta, g_inh, Iapp):
    asyn = g_inh * np.ones((N,N))
    np.fill_diagonal(asyn, 0.0)
    def network(t, S):
        dS = []
        Vs = np.array([S[i*4+2] for i in range(N)])
        Isyn = asyn @ sigmoid(Vs)
        for i in range(N):
            dS.extend(neuron_eqs(S[i*4:(i+1)*4], Iapp + Isyn[i], alpha, delta, 1.0, 50.0, 2500.0))
        return dS
    return network


def run_cpg(N=4, tmax=80000):
    alpha = [-2.0,2.0,-1.5,1.5]
    delta = [0.0,0.0,-1.5,-1.5]
    g_inh, Iapp = -0.3, -1.6
    S0 = np.zeros(N*4)
    S0[::4] = -1.0 + 0.1*np.random.randn(N)
    t_eval = np.arange(0, tmax, 1)
    sol = solve_ivp(make_network(N, alpha, delta, g_inh, Iapp), (0,tmax), S0, t_eval=t_eval, method="BDF")
    
    # Decode phase from spikes
    spike_thresh = -2.0
    prev_vm = sol.y[0::4,0].copy()
    spike_times, decoded_angles = [], []
    phases = np.linspace(0, 2*np.pi, N, endpoint=False)
    for k, t_now in enumerate(sol.t):
        for i in range(N):
            vm = sol.y[i*4,k]
            if vm > spike_thresh and prev_vm[i] <= spike_thresh:
                theta = np.degrees(np.arctan2(np.sin(phases[i]), np.cos(phases[i]))) % (360)
                spike_times.append(t_now)
                decoded_angles.append(theta)
            prev_vm[i] = vm
    spike_times = np.array(spike_times)
    decoded_angles = np.array(decoded_angles)
    
    if len(spike_times) > 4:
        t_uniform = np.linspace(spike_times.min(), spike_times.max(), 2000)
        theta_interp = np.interp(t_uniform, spike_times, decoded_angles)
        theta_smooth = gaussian_filter1d(theta_interp, sigma=8.0)
    else:
        t_uniform = sol.t
        theta_smooth = np.sin(2*np.pi*t_uniform/tmax)*180 + 180
    return sol, t_uniform, theta_smooth

sol, t_cpg, theta_smooth = run_cpg()


plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(sol.t, sol.y[i*4,:], label=f'Neuron {i+1} Vm')
plt.title("CPG Membrane Potentials (Vm)")
plt.xlabel("Time")
plt.ylabel("Vm")
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t_cpg, theta_smooth, color='orange')
plt.title("Decoded Phase θ (smoothed)")
plt.xlabel("Time")
plt.ylabel("θ (degrees)")
plt.show()

def build_dataset(theta, gait_array, gait_id, seq_len=50):
    th_norm = (theta - theta.min())/(theta.max()-theta.min())*2 - 1
    gait_norm = (gait_array - gait_array.min())/(gait_array.max()-gait_array.min())*2 - 1
    X, y = [], []
    for i in range(len(th_norm)-seq_len):
        seq = th_norm[i:i+seq_len]
        gait_channel = np.ones((seq_len,1))*gait_id
        seq2d = np.concatenate([seq.reshape(seq_len,1), gait_channel], axis=1)
        X.append(seq2d)
        y.append(gait_norm[i % gait_norm.shape[0]])
    return np.array(X), np.array(y)

Xw, yw = build_dataset(theta_smooth, wkF, 0)
Xc, yc = build_dataset(theta_smooth, crF, 1)
X = np.vstack([Xw,Xc])
y = np.vstack([yw,yc])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def build_rnn(seq_len=50, input_dim=2, output_dim=8):
    inp = layers.Input(shape=(seq_len, input_dim))
    x = layers.SimpleRNN(128, activation='tanh')(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(output_dim, activation='linear')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_rnn()
model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test,y_test), verbose=2)


y_pred = model.predict(X_test)
plt.figure(figsize=(10,5))
plt.plot(y_test[:500,0], label='True (joint0)')
plt.plot(y_pred[:500,0], label='Pred (joint0)')
plt.title("RNN Joint Angle Prediction Example")

plt.figure(figsize=(10,5))
plt.plot(y_test[:500,0], label='True (joint3)')
plt.plot(y_pred[:500,0], label='Pred (joint3)')
plt.title("RNN Joint Angle Prediction Example")

plt.figure(figsize=(10,5))
plt.plot(y_test[:500,0], label='True (joint5)')
plt.plot(y_pred[:500,0], label='Pred (joint5)')
plt.title("RNN Joint Angle Prediction Example")

plt.figure(figsize=(10,5))
plt.plot(y_test[:500,0], label='True (joint7)')
plt.plot(y_pred[:500,0], label='Pred (joint7)')
plt.title("RNN Joint Angle Prediction Example")
plt.legend()
plt.show()

def visualize_gait(model, theta_smooth, gait_id, title):
    seq_len = 50
    gait_channel = np.ones((seq_len,1))*gait_id
    th_norm = (theta_smooth - theta_smooth.min())/(theta_smooth.max()-theta_smooth.min())*2 - 1
    preds = []
    for i in range(len(th_norm)-seq_len):
        seq = th_norm[i:i+seq_len].reshape(seq_len,1)
        seq2d = np.concatenate([seq,gait_channel], axis=1)[None,...]
        y_pred = model.predict(seq2d, verbose=0)[0]
        preds.append(y_pred)
    preds = np.array(preds)
    plt.figure(figsize=(10,5))
    for j in range(8):
        plt.plot(preds[:,j], label=f"Joint{j}")
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Normalized angle")
    plt.legend(ncol=4)
    plt.show()

visualize_gait(model, theta_smooth, gait_id=0, title="Learned WALK gait")
visualize_gait(model, theta_smooth, gait_id=1, title="Learned CRAWL gait")


model.save("rnn_gait_decoder.keras")
with open("rnn_cpg_scaler.pkl","wb") as f:
    pickle.dump((wkF.min(), wkF.max()), f)
print(" Model and scaling info saved")

with open("rnn2_gait_decoder_arch.json", "w") as f:
    f.write(model.to_json())

model.save_weights("rnn_gait_decoder.weights.h5")

print(" Training complete. Model files saved.")
