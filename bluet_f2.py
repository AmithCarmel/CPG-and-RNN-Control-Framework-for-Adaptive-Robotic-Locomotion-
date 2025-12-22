import bluetooth
import numpy as np
import tensorflow as tf
import time
import os
import sys
import shutil
import matplotlib.pyplot as plt
from collections import deque

SEQ_LEN = 50
CONTROL_FREQ_HZ = 10.0
BLUETOOTH_MAC_ADDRESS = 'FC:B4:67:20:F2:6E'
BLUETOOTH_PORT = 1
GAIT_WALK = "wkF"
GAIT_CRAWL = "crF"
THRESHOLD = -0.176
BAR_WIDTH = 20

class GaitDecoderBluetooth:
    def __init__(self):
        self.time_step = 1.0 / CONTROL_FREQ_HZ
        current_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            with open(os.path.join(current_dir, 'rnn_gait_decoder_arch.json'), 'r') as f:
                self.model = tf.keras.models.model_from_json(f.read())
            self.model.load_weights(os.path.join(current_dir, 'rnn_gait_decoder.weights.h5'))
            print("RNN loaded")
        except Exception as e:
            print("RNN load failed:", e)
            sys.exit(1)
        self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
        try:
            self.sock.connect((BLUETOOTH_MAC_ADDRESS, BLUETOOTH_PORT))
            time.sleep(1)
            print("Bluetooth connected")
        except Exception as e:
            print("Bluetooth failed:", e)
            sys.exit(1)
        self.theta_buffer = np.zeros((SEQ_LEN, 2))
        self.theta_time = 0.0
        self.theta_hist = deque(maxlen=200)
        self.walk_hist = deque(maxlen=200)
        self.crawl_hist = deque(maxlen=200)
        plt.ion()
        self.fig, (self.ax_theta, self.ax_walk, self.ax_crawl) = plt.subplots(3, 1, sharex=True)
        self.ax_theta.set_title("Theta")
        self.ax_walk.set_title("Walk gait")
        self.ax_crawl.set_title("Crawl gait")
        self.theta_line, = self.ax_theta.plot([], [])
        self.walk_lines = []
        self.crawl_lines = []
        self.num_joints = None

    def _get_theta(self):
        self.theta_time += self.time_step
        return np.sin(2 * np.pi * self.theta_time / 2.0)

    def _print_status(self, theta, activation, gait, cmd):
        filled = int(max(0.0, min(1.0, activation)) * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        cols = shutil.get_terminal_size().columns
        line = f"θ={theta:+.2f} | RNN={activation:+.3f} | GAIT={gait} | CMD={cmd} | {bar}"
        print("\r" + line[:cols], end="", flush=True)

    def _init_joint_plots(self, num_joints):
        for _ in range(num_joints):
            wl, = self.ax_walk.plot([], [])
            cl, = self.ax_crawl.plot([], [])
            self.walk_lines.append(wl)
            self.crawl_lines.append(cl)
        self.ax_walk.set_ylim(-1.5, 1.5)
        self.ax_crawl.set_ylim(-1.5, 1.5)
        self.num_joints = num_joints

    def _update_plots(self):
        x = np.arange(len(self.theta_hist))
        self.theta_line.set_data(x, list(self.theta_hist))
        self.ax_theta.set_ylim(-1.2, 1.2)
        self.ax_theta.set_xlim(0, max(50, len(x)))
        if self.num_joints is not None:
            if len(self.walk_hist) > 0:
                walk_arr = np.array(self.walk_hist)
                xw = np.arange(len(walk_arr))
                for j in range(self.num_joints):
                    self.walk_lines[j].set_data(xw, walk_arr[:, j])
                self.ax_walk.set_xlim(0, max(50, len(xw)))
            if len(self.crawl_hist) > 0:
                crawl_arr = np.array(self.crawl_hist)
                xc = np.arange(len(crawl_arr))
                for j in range(self.num_joints):
                    self.crawl_lines[j].set_data(xc, crawl_arr[:, j])
                self.ax_crawl.set_xlim(0, max(50, len(xc)))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def send_cmd(self, cmd):
        print(f"\n> {cmd}")
        self.sock.send((cmd.strip() + "\n").encode())

    def run(self):
        self.send_cmd("k up")
        time.sleep(1)
        try:
            while True:
                theta = self._get_theta()
                self.theta_hist.append(theta)
                self.theta_buffer[:-1] = self.theta_buffer[1:]
                self.theta_buffer[-1] = [theta, 1.0]
                rnn_input = self.theta_buffer.reshape(1, SEQ_LEN, 2)
                rnn_out = self.model.predict(rnn_input, verbose=0)[0]
                if self.num_joints is None:
                    self._init_joint_plots(len(rnn_out))
                activation = float(np.mean(rnn_out))
                gait = GAIT_WALK if activation > THRESHOLD else GAIT_CRAWL
                cycles = 5 if gait == GAIT_WALK else 3
                cmd = f"k {gait} {cycles}"
                self.send_cmd(cmd)
                if gait == GAIT_WALK:
                    self.walk_hist.append(rnn_out.copy())
                else:
                    self.crawl_hist.append(rnn_out.copy())
                self._update_plots()
                self._print_status(theta, activation, gait, cmd)
                time.sleep(cycles * 0.8)
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.send_cmd("k up")
            self.sock.close()

if __name__ == "__main__":
    GaitDecoderBluetooth().run()
