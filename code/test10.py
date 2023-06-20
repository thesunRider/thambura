from rtlsdr import RtlSdr
import numpy as np

sample_rate = 3.2e6
duration = 3

sdr = RtlSdr()

# some defaults
sdr.rs = sample_rate
sdr.fc = 433e6
sdr.gain = 10

samples = sdr.read_samples(sample_rate*duration)

np.save('data.npy', samples) # save
print(samples)
sdr.close()