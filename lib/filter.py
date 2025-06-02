import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi, find_peaks
import wfdb

# Parameters
fs = 360  # Sampling frequency
duration_sec = 100
n_samples = fs * duration_sec

# Step 1: Load ECG
record = wfdb.rdrecord('mitdb/100', sampto=n_samples)
ecg = record.p_signal[:, 0]

# Step 2: Load annotations
ann = wfdb.rdann('mitdb/100', 'atr', sampto=n_samples)
ann_sample = np.array(ann.sample)
ann_sample = ann_sample[ann_sample < len(ecg)]

# Step 3: Design bandpass filter directly for 360 Hz
lowcut = 0.5
highcut = 40.0
order = 4

nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq

b, a = butter(order, [low, high], btype='bandpass')
#b = [0.41917317, 0.46722821, 0.52597928, 0.00967556, 0.13042169, 0.39705387,
#0.65865885 ,0.19454516, 0.72859411]
#a = [0.91607871, 0.67942481, 0.68628766, 0.93210451, 0.87962741, 0.53466653,
#     0.41345249, 0.23527293, 0.24730684]

print(b)
print(a)
# Step 4: Apply filter
zi = lfilter_zi(b, a) * ecg[0]
filtered_ecg, _ = lfilter(b, a, ecg, zi=zi)

# Step 5: Peak detection
min_distance = int(0.6 * fs)
filtered_peaks, _ = find_peaks(filtered_ecg, distance=min_distance, prominence=0.05)

# Step 6: Match annotations
tolerance = 30
matched_ann = []
matched_detected = []

for ref in ann_sample:
    diffs = np.abs(filtered_peaks - ref)
    if np.any(diffs <= tolerance):
        matched_idx = np.argmin(diffs)
        matched_ann.append(ref)
        matched_detected.append(filtered_peaks[matched_idx])

# Step 7: Print matching info
print("Matched Peaks (Redesigned Filter @ 360 Hz):")
for gt, det in zip(matched_ann, matched_detected):
    print(f"Annotation Index: {gt}, Detected Index: {det}, Time Error: {(det - gt) / fs:.3f} s")

# Step 8: Plot
start = 10000
end = start + fs * 10
time = np.arange(start, end) / fs

plt.figure(figsize=(12, 5))
plt.plot(time, ecg[start:end], label='Raw ECG (360 Hz)', alpha=0.5)
plt.plot(time, filtered_ecg[start:end], label='Filtered ECG (Butter Bandpass)', linewidth=1.5)

visible_ann = ann_sample[(ann_sample >= start) & (ann_sample < end)]
visible_peaks = filtered_peaks[(filtered_peaks >= start) & (filtered_peaks < end)]

plt.plot(visible_ann / fs, ecg[visible_ann], 'rx', label='Annotations')
plt.plot(visible_peaks / fs, filtered_ecg[visible_peaks], 'go', label='Detected Peaks')

plt.title("Redesigned Bandpass Filter for ECG (360 Hz) with Annotations")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
