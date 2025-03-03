import mne
import numpy as np
import matplotlib.pyplot as plt

# Define the file path
file_path = r"C:\Users\elbuc\Documents\Personal Projects\Coding\Clean_Data\007\007_EEG.csv"

# Load the CSV file using numpy
data = np.loadtxt(file_path, delimiter=',', skiprows=1)  # Assuming the first row is headers

# Extract only the third EEG column and reshape
eeg_data = data[:, 2].reshape(-1, 1)

# Define channel names
ch_names = ["EEG_3"]

# Sampling frequency (adjust according to your dataset)
sfreq = 512  # Hz
time = np.arange(len(eeg_data)) / sfreq  # Create a time axis in seconds

# Create MNE Info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

# Create MNE Raw object
raw = mne.io.RawArray(eeg_data.T, info)

# Apply Bandpass Filtering (5Hz - 40Hz)
raw_filtered = raw.copy().filter(l_freq=5, h_freq=40, fir_design='firwin')

# Apply Notch Filtering at Multiples of 8Hz (8, 16, 24, 32, 40)
notch_freqs = np.arange(8, 41, 8)  # Generates [8, 16, 24, 32, 40]
raw_filtered = raw_filtered.notch_filter(freqs=notch_freqs)

# Compute Power Spectral Density (PSD)
psd_before = raw.compute_psd(method="welch", fmin=1, fmax=60, n_fft=1024)
psd_after = raw_filtered.compute_psd(method="welch", fmin=1, fmax=60, n_fft=1024)

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Plot before filtering
axs[0].semilogy(f, psd_unfiltered[0], color='b', label="Before Filtering")
axs[0].set_ylabel("Power (dB/Hz)")
axs[0].set_title("Power Spectral Density Before Filtering")
axs[0].legend()
axs[0].grid()

# Plot after filtering
axs[1].semilogy(f_filtered, psd_filtered[0], color='r', label="After Filtering (5-40Hz + Notch)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Power (dB/Hz)")
axs[1].set_title("Power Spectral Density After Filtering")
axs[1].legend()
axs[1].grid()

# Adjust layout
plt.tight_layout()
plt.show()
stop=1