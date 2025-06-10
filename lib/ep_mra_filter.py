
import matplotlib
matplotlib.use('QtAgg')  # Use Qt GUI backend

import numpy as np
import pywt
import wfdb
from scipy.signal import lfilter, lfilter_zi, find_peaks
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PyQt5 import QtWidgets
import sys

# Load ECG and annotations
record = wfdb.rdrecord('mitdb/100')
signal = record.p_signal[:, 0]
fs = 360
time = np.arange(len(signal)) / fs

# EP-optimized filter
wavelet_name = 'coif1'
subbands_included = [1, 0, 1, 1]
b = np.array([0.89615792, 0.0498606, 0.29551737, 0.1159615, 0.67357345,
              0.28109583, 0.96102963, 0.00411725, 0.05462859])
a = np.array([0.67769491, 0.26560846, 0.40112539, 0.18727869, 0.54770744,
              0.17837254, 0.48446466, 0.15923269, 0.0564932])

# Wavelet + IIR Filtering
coeffs = pywt.wavedec(signal, wavelet_name, level=4)
coeffs = [c if i == 0 or subbands_included[i - 1] else np.zeros_like(c) for i, c in enumerate(coeffs)]
denoised = pywt.waverec(coeffs, wavelet_name)[:len(signal)]
denoised = np.nan_to_num(denoised, nan=0.0, posinf=5.0, neginf=-5.0)
zi = lfilter_zi(b, a) * denoised[0]
filtered, _ = lfilter(b, a, denoised, zi=zi)
filtered = np.nan_to_num(filtered[:len(signal)], nan=0.0, posinf=5.0, neginf=-5.0)

# Peak detection
min_distance = int(0.6 * fs)
raw_peaks, _ = find_peaks(signal, distance=min_distance, prominence=0.2)
filt_peaks, _ = find_peaks(filtered, distance=min_distance, prominence=0.2)

# Initial window
window_size = fs * 10
start = 0
end = start + window_size

# Plot with slider
fig, ax = plt.subplots(figsize=(12, 5))
plt.subplots_adjust(bottom=0.25)
l_raw, = ax.plot(time[start:end], signal[start:end], label='Original', alpha=0.6)
l_filt, = ax.plot(time[start:end], filtered[start:end], label='Filtered', linewidth=1)
raw_marks, = ax.plot(time[raw_peaks[(raw_peaks >= start) & (raw_peaks < end)]],
                     signal[raw_peaks[(raw_peaks >= start) & (raw_peaks < end)]],
                     'rx', label='Raw R-peaks')
filt_marks, = ax.plot(time[filt_peaks[(filt_peaks >= start) & (filt_peaks < end)]],
                      filtered[filt_peaks[(filt_peaks >= start) & (filt_peaks < end)]],
                      'go', label='Filtered R-peaks')

ax.set_xlim(time[start], time[end - 1])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (mV)')
ax.set_title("Scrollable ECG Viewer with R-Peak Comparison (QtAgg)")
ax.legend()
ax.grid(True)

# Slider
slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(slider_ax, 'Start Time (s)', 0, len(signal) / fs - 10, valinit=0, valstep=1)

def update(val):
    start = int(val * fs)
    end = start + window_size
    l_raw.set_xdata(time[start:end])
    l_raw.set_ydata(signal[start:end])
    l_filt.set_xdata(time[start:end])
    l_filt.set_ydata(filtered[start:end])
    raw_seg = (raw_peaks >= start) & (raw_peaks < end)
    filt_seg = (filt_peaks >= start) & (filt_peaks < end)
    raw_marks.set_xdata(time[raw_peaks[raw_seg]])
    raw_marks.set_ydata(signal[raw_peaks[raw_seg]])
    filt_marks.set_xdata(time[filt_peaks[filt_seg]])
    filt_marks.set_ydata(filtered[filt_peaks[filt_seg]])
    ax.set_xlim(time[start], time[end - 1])
    fig.canvas.draw_idle()

slider.on_changed(update)

# Run in Qt event loop
app = QtWidgets.QApplication(sys.argv)
fig.show()
app.exec_()
