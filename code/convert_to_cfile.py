import numpy as np

#data2_non is pure noise,,data has signal

sig = np.load('data.npy')
data = np.array([sig.real,sig.imag]).T
(data.flatten().astype(np.float32)).tofile("data.cfile")