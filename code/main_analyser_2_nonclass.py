import numpy as np
from scipy.signal import butter, lfilter, freqz, resample,resample_poly , filtfilt,decimate
import time,threading

sf = 7;            # spreading factor
bw = 125e3;         # bandwidth
fs = 3.2e6;           # sampling rate
cfo = 0;
fast_mode = False

sig = None

zero_padding_ratio = 10
sample_num = 2 * 2**sf;
preamble_len = 8
bin_num = 2**sf * zero_padding_ratio
fft_len = sample_num*zero_padding_ratio;
    

def lowpass(data, cutoff, fs, order=1):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False) 
    y = filtfilt(b, a, data)
    return y

def dechirp(x):
    # dechirp  Apply dechirping on the symbol starts from index x
    #
    # input:
    #     x: Start index of a symbol
    #     is_up: `true` if applying up-chirp dechirping
    #            `false` if applying down-chirp dechirping
    # output:
    #     pk: Peak in FFT results of dechirping
    #         pk = (height, index)

    c = downchirp;
    ft = np.fft.fft(np.multiply(sig[x-1:x+sample_num-1],c), fft_len);
    ft_ = np.abs(ft[0:bin_num]) + abs(ft[fft_len-bin_num:fft_len]);
    pk = topn(np.array([ft_[0:bin_num].T]), 1)
    return pk[0]


def chirp(is_up, sf, bw, fs, h, cfo=0, tdelta=0, tscale=1):
    """
    Generate a LoRa chirp symbol

    Args:
    - is_up: `True` if constructing an up-chirp, `False` if constructing a down-chirp
    - sf: Spreading Factor
    - bw: Bandwidth
    - fs: Sampling Frequency
    - h: Start frequency offset (0 to 2^SF-1)
    - cfo: Carrier Frequency Offset (default: 0)
    - tdelta: Time offset (0 to 1/fs) (default: 0)
    - tscale: Scaling the sampling frequency (default: 1)

    Returns:
    - y: Generated LoRa symbol
    """

    if tscale is None:
        tscale = 1
    if tdelta is None:
        tdelta = 0
    if cfo is None:
        cfo = 0

    N = 2 ** sf
    T = N / bw
    samp_per_sym = round(fs / bw * N)
    h_orig = h
    h = round(h)
    cfo = cfo + (h_orig - h) / N * bw

    if is_up:
        k = bw / T
        f0 = -bw / 2 + cfo
    else:
        k = -bw / T
        f0 = bw / 2 + cfo

    # retain last element to calculate phase
    t = (np.arange(samp_per_sym * (N - h) // N + 1) / fs * tscale) + tdelta
    c1 = np.exp(1j * 2 * np.pi * (t * (f0 + k * T * h / N + 0.5 * k * t)))

    if len(c1) == 0:
        phi = 0
    else:
        phi = np.angle(c1[-1])
    t = np.arange(samp_per_sym * h // N) / fs + tdelta
    c2 = np.exp(1j * (phi + 2 * np.pi * (t * (f0 + 0.5 * k * t))))

    y = np.concatenate((c1[:-1], c2)).T
    return y

def topn(pks, n, padding=False, th=None):
    pks = np.append(np.array(pks),np.array([np.arange(len(pks[0]))]),axis=0).T
    y = pks[pks[:,0].argsort()[::-1]][:,0]
    p = pks[pks[:,0].argsort()[::-1]][:,1]
    if n is None:
        return y, p
    nn = min(n, pks.shape[0])
    if padding:
        y = np.vstack([pks[p[:nn], :], np.zeros((n - nn, pks.shape[1]))])
    else:
        y = pks[int((p[:nn])[0])]
    if th is not None:
        ii = 0
        while ii < y.shape[0] and abs(y[ii, 0]) >= th:
            ii += 1
        y = y[:ii, :]
    return y, p[:nn]



def detect(start_idx):
    #detect  Detect preamble
    #
    #input:
    #     start_idx: Start index for detection
    # output:
    #     x: Before index x, a preamble is detected.
    #        x = -1 if no preamble detected

    ii = start_idx; 
    pk_bin_list = np.array([]); # preamble peak bin list
    while ii < len(sig) - sample_num * preamble_len :
        #search preamble_len-1 basic upchirps
        if len(pk_bin_list) == preamble_len - 1:
            # preamble detected
            # coarse alignment: first shift the up peak to position 0
            # current sampling frequency = 2 * bandwidth
            x = ii - round((pk_bin_list[-1]-1)/zero_padding_ratio*2)
            return x

        pk0 = dechirp(ii)
        if len(pk_bin_list) > 0:
            bin_diff = (pk_bin_list[-1]-pk0[1])% bin_num

            if bin_diff > bin_num/2:
                bin_diff = bin_num - bin_diff

            if bin_diff <= zero_padding_ratio:
                pk_bin_list = np.append(pk_bin_list,pk0[1])
            else:
                pk_bin_list = np.array([pk0[1]])

        else:
            pk_bin_list = np.array([pk0[1]])

        ii = ii + sample_num

    x = -1
    return x




with open("res\\sig.cfile","rb") as fid:
    signal_ar = np.fromfile(fid, np.float32).reshape((-1, 2)).T



signal = np.load("data.npy")#signal_ar[0,:]+ 1j * signal_ar[1,:]


for i in range(5,510):
    bw = i * (10**3)
    sig = signal
    downchirp = chirp(False, sf, bw, 2*bw, 0, cfo, 0);
    if not fast_mode :
        sig = lowpass(sig, bw/2, fs);

    sig =  resample_poly(sig, 2*bw, fs)

    start3 = time.time()
    x = detect(1)



