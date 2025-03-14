import mne
import numpy as np
import matplotlib.pyplot as plt
import os

# Convert integer to a zero-padded string

# Define the file path
file_path = fr"C:\Users\elbuc\Documents\Personal Projects\Coding\Clean_Data\007\007_EEG.csv"

# Assuming the first row is headers
# Load the data with numpy, treating empty strings as NaNs
# Load the CSV file with numpy, treating empty strings as NaNs
data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype='float64', missing_values='', filling_values=np.nan)

# Remove rows with any NaN values (rows that had empty strings)
data = data[~np.isnan(data).any(axis=1)]

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

# Extract data for plotting
eeg_before = raw.get_data()[0]  # Get the first (and only) channel's data before filtering
eeg_after = raw_filtered.get_data()[0]  # Get the same channel's data after filtering

# Compute PSD for raw and filtered EEG data
psd_before = raw.compute_psd(fmax=60)
psd_after = raw_filtered.compute_psd(fmax=60)

# Extract frequency values and PSD data
freqs_before = psd_before.freqs
freqs_after = psd_after.freqs

psd_before_values = psd_before.get_data()
psd_after_values = psd_after.get_data()

# Create a figure with two windows
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

# Plot before filtering
axs[0, 0].plot(time, eeg_before, color='b', label="Before Filtering")
axs[0, 0].set_ylabel("EEG Amplitude (µV)")
axs[0, 0].set_title("EEG Signal Before Filtering")
axs[0, 0].legend()
axs[0, 0].grid()

# Plot after filtering
axs[0, 1].plot(time, eeg_after, color='r', label="After Filtering (5-40Hz + Notch)")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("EEG Amplitude (µV)")
axs[0, 1].set_title("EEG Signal After Filtering")
axs[0, 1].legend()
axs[0, 1].grid()

# Plot PSD Before Filtering
axs[1, 0].plot(freqs_before, psd_before_values.mean(axis=0), label="Before Filtering", color='b')
axs[1, 0].set_xlabel("Frequency (Hz")
axs[1, 0].set_ylabel("Power (µV^2/Hz)")
axs[1, 0].set_title("PSD Before Filtering")
axs[1, 0].grid()

# Plot PSD After Filtering
axs[1, 1].plot(freqs_after, psd_after_values.mean(axis=0), label="After Filtering", color='r')
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_ylabel("Power (µV^2/Hz)")
axs[1, 1].set_title("PSD After Filtering (5-40Hz + Notch)")
axs[1, 1].grid()

# Save the figure
save_dir = r"C:\Users\elbuc\Documents\Personal Projects\Coding\EEG Results\Channel_3"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "007_EEG_003.jpg")
plt.tight_layout()
plt.savefig(save_path, format='jpg')
plt.close()

print(f"Saved: {save_path}")