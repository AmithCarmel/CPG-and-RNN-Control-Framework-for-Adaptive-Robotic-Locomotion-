import bluetooth
import numpy as np
import tensorflow as tf
import time
import os
import sys
import matplotlib.pyplot as plt
from collections import deque

SEQ_LEN = 50
CONTROL_HZ = 10.0
DT = 1.0 / CONTROL_HZ

BLUETOOTH_MAC_ADDRESS = 'FC:B4:67:20:F2:6E'
BLUETOOTH_PORT = 1

GAIT_WALK = 0
GAIT_CRAWL = 1

Tf, Ts, Tus = 1.0, 50.0, 2500.0
alpha = [-2.0, 2.0, -1.5, 1.5]
delta = [0.0, 0.0, -1.5, -1.5]
Iapp = -1.6

class CPG:
    def __init__(self):
        self.S = np.zeros(4)
        self.S[0] = -1.0

    def step(self):
        vm, vf, vs, vus = self.S

        dvm = (
            -vm
            - alpha[0] * np.tanh(vf - delta[0])
            - alpha[1] * np.tanh(vs - delta[1])
            - alpha[2] * np.tanh(vs - delta[2])
            - alpha[3] * np.tanh(vus - delta[3])
            + Iapp
        )

        dvf = (vm - vf) / Tf
        dvs = (vm - vs) / Ts
        dvus = (vm - vus) / Tus

        self.S += DT * np.array([dvm, dvf, dvs, dvus])

        theta = self.S[2]   # v_s = PHASE
        return theta

class GaitController:
    def __init__(self):
        # ---- Load RNN ----
        with open("rnn_gait_decoder_arch.json", "r") as f:
            self.model = tf.keras.models.model_from_json(f.read())
        self.model.load_weights("rnn_gait_decoder.weights.h5")
        print("✅ RNN loaded")

        # ---- Bluetooth ----
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        self.sock.connect((BLUETOOTH_MAC_ADDRESS, BLUETOOTH_PORT))
        print("✅ Bluetooth connected")

        # ---- CPG ----
        self.cpg = CPG()

        # ---- Buffers ----
        self.theta_buffer = np.zeros((SEQ_LEN, 2))
        self.gait_id = GAIT_WALK

        self.theta_hist = deque(maxlen=300)
        self.joint_hist = deque(maxlen=300)

        # ---- Plots ----
        plt.ion()
        self.fig, (self.ax_theta, self.ax_joint) = plt.subplots(2, 1, figsize=(10, 6))

        self.theta_line, = self.ax_theta.plot([], [], 'm')
        self.joint_lines = []

        self.num_joints = None


    def send_cmd(self, cmd):
        self.sock.send((cmd + "\n").encode())

    def init_joint_plot(self, n):
        for _ in range(n):
            line, = self.ax_joint.plot([], [])
            self.joint_lines.append(line)
        self.num_joints = n


    def update_plots(self):
        x = np.arange(len(self.theta_hist))

        # Theta plot
        self.theta_line.set_data(x, self.theta_hist)
        self.ax_theta.set_ylim(-2, 2)
        self.ax_theta.set_xlim(0, max(100, len(x)))
        self.ax_theta.set_title("CPG Phase θ(t)")

        # Joint plot
        if self.num_joints:
            J = np.array(self.joint_hist)
            for j in range(self.num_joints):
                self.joint_lines[j].set_data(x, J[:, j])
            self.ax_joint.set_ylim(-1.2, 1.2)
            self.ax_joint.set_xlim(0, max(100, len(x)))
            self.ax_joint.set_title(
                "Joint Outputs (" + ("WALK" if self.gait_id == 0 else "CRAWL") + ")"
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        print(" Running controller (Ctrl+C to stop)")
        print(" WALK = gait_id 0 | CRAWL = gait_id 1")

        try:
            while True:
                if int(time.time()) % 20 < 10:
                    self.gait_id = GAIT_WALK
                else:
                    self.gait_id = GAIT_CRAWL

                
                theta = self.cpg.step()
                self.theta_hist.append(theta)

                
                self.theta_buffer[:-1] = self.theta_buffer[1:]
                self.theta_buffer[-1] = [theta, self.gait_id]

                rnn_input = self.theta_buffer[None, ...]
                rnn_out = self.model.predict(rnn_input, verbose=0)[0]

                if self.num_joints is None:
                    self.init_joint_plot(len(rnn_out))

                self.joint_hist.append(rnn_out.copy())

		
                gait_str = "wkF" if self.gait_id == 0 else "crF"
                self.send_cmd(f"k {gait_str} 1")

                
                self.update_plots()

                time.sleep(DT)

        except KeyboardInterrupt:
            print("\n Stopped")
        finally:
            self.send_cmd("k up")
            self.sock.close()


if __name__ == "__main__":
    GaitController().run()

