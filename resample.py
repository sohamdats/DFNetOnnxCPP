import torchaudio
import torch

# Load an example audio file (replace this with your actual file)
waveform, sample_rate = torchaudio.load('Kal_Ho_na_ho_Deb.wav')

# Define the original and new sample rates
orig_freq = sample_rate  # Original sample rate of the audio file
new_freq = 48000         # New sample rate you want to resample to

# Create a resampler
resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)

# Resample the waveform
resampled_waveform = resampler(waveform)

# Save the resampled waveform to a new file (optional)
torchaudio.save('Kal_Ho_na_ho_Deb.wav', resampled_waveform, new_freq)

# Print some information about the original and resampled waveforms
print(f'Original waveform shape: {waveform.shape}, Sample rate: {orig_freq}')
print(f'Resampled waveform shape: {resampled_waveform.shape}, Sample rate: {new_freq}')
