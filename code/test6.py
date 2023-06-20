from LoraPhy import LoRaPHY
import numpy as np
rf_freq = 470e6;    # carrier frequency, used to correct clock drift

sf = 7;             # spreading factor
bw = 500e3;         # bandwidth
fs = 1e6;           # sampling rate

phy = LoRaPHY(rf_freq, sf, bw, fs)


# Encode payload [1 2 3 4 5]

symbols = phy.encode(np.array([1,2,3,4,5]).T)

# Baseband Modulation
#sig = phy.modulate(symbols);