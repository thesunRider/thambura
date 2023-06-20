import numpy as np
from scipy.signal import butter, lfilter, freqz, resample,resample_poly , filtfilt
import time 
from threading import Thread
from multiprocessing import Pool, RawArray

var_dict  = {}

class CAD_LORA(object):
    sf = 7;            # spreading factor
    bw = 125e3;         # bandwidth
    fs = 3.2e6;           # sampling rate
    cfo = 0;
    fast_mode = False
    sf_range = (7,12)
    sig = None

    """Channel Activity Detetcion for LORA"""
    def __init__(self, sig,sf,bw,fs,cfo = 0, fast_mode= False,detect_sf = False):
        super(CAD_LORA, self).__init__()
        self.sig = sig
        self.signal = sig.copy()
        self.detect_sf = detect_sf
        self.fast_mode=fast_mode
        self.setup_params(sf,bw,fs,cfo)

    def setup_params(self,sf,bw,fs,cfo = 0):
        self.sf = sf
        self.bw = bw
        self.fs = fs
        self.cfo = cfo

        self.zero_padding_ratio = 10
        self.sample_num = 2 * 2**self.sf;
        self.preamble_len = 8
        self.bin_num = 2**self.sf * self.zero_padding_ratio
        self.fft_len = self.sample_num*self.zero_padding_ratio;
        

    def lowpass(self,data, cutoff, fs, order=1):
        b, a = butter(order, cutoff, fs=fs, btype='low', analog=False) 
        y = filtfilt(b, a, data)
        return y

    def dechirp(self,x):
        # dechirp  Apply dechirping on the symbol starts from index x
        #
        # input:
        #     x: Start index of a symbol
        #     is_up: `true` if applying up-chirp dechirping
        #            `false` if applying down-chirp dechirping
        # output:
        #     pk: Peak in FFT results of dechirping
        #         pk = (height, index)

        c = self.downchirp;
        #breakpoint()
        ft = np.fft.fft(np.multiply(self.sig[x-1:x+self.sample_num-1],c), self.fft_len);
        ft_ = np.abs(ft[0:self.bin_num]) + abs(ft[self.fft_len-self.bin_num:self.fft_len]);
        pk = self.topn(np.array([ft_[0:self.bin_num].T]), 1)
        return pk[0]


    def chirp(self,is_up, sf, bw, fs, h, cfo=0, tdelta=0, tscale=1):
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

    def topn(self,pks, n, padding=False, th=None):
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



    def detect(self,start_idx):
        #detect  Detect preamble
        #
        #input:
        #     start_idx: Start index for detection
        # output:
        #     x: Before index x, a preamble is detected.
        #        x = -1 if no preamble detected

        ii = start_idx; 
        pk_bin_list = np.array([]); # preamble peak bin list
        while ii < len(self.sig) - self.sample_num * self.preamble_len :
            #search preamble_len-1 basic upchirps
            if len(pk_bin_list) == self.preamble_len - 1:
                # preamble detected
                # coarse alignment: first shift the up peak to position 0
                # current sampling frequency = 2 * bandwidth
                x = ii - round((pk_bin_list[-1]-1)/self.zero_padding_ratio*2)
                return x

            pk0 = self.dechirp(ii)
            if len(pk_bin_list) > 0:
                bin_diff = (pk_bin_list[-1]-pk0[1])% self.bin_num

                if bin_diff > self.bin_num/2:
                    bin_diff = self.bin_num - bin_diff

                if bin_diff <= self.zero_padding_ratio:
                    pk_bin_list = np.append(pk_bin_list,pk0[1])
                else:
                    pk_bin_list = np.array([pk0[1]])

            else:
                pk_bin_list = np.array([pk0[1]])

            ii = ii + self.sample_num

        x = -1
        return x



    def find_preamble(self,x=1):
        self.sig = self.signal

        start2 = time.time()
        self.downchirp = self.chirp(False, self.sf, self.bw, 2*self.bw, 0, self.cfo, 0);
        if not self.fast_mode :
            self.sig = self.lowpass(self.sig, self.bw/2, self.fs);
        self.sig =  resample_poly(self.sig, 2*self.bw, self.fs)

        if self.detect_sf:
            array_x = np.array([])
            for i in range(self.sf_range[0],self.sf_range[1]):
                self.setup_params(i,self.bw,self.fs,self.cfo)
                self.downchirp = self.chirp(False, i, self.bw, 2*self.bw, 0, self.cfo, 0);
                x = self.detect(1)
                array_x = np.append(array_x,(i,x))
            return array_x
        else:
                x = self.detect(x)
                return x
    

def calc_preamble_multi(bw):
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    sig = X_np[0] + 1j*X_np[1]
    cad_lora = CAD_LORA(sig,7,bw,3.2E6,0,detect_sf = True)
    found_x = cad_lora.find_preamble()
    print("Used bw:",bw,"X:",found_x)
    return found_x

def calc_preamble(sig,bw):
    cad_lora = CAD_LORA(sig,7,bw,3.2E6,0,detect_sf = True)
    found_x = cad_lora.find_preamble()
    print("Used bw:",bw,"X:",found_x)
    return found_x

def init_worker(X, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape

if __name__ == '__main__':   
    #with open("res\\sig.cfile","rb") as fid:
    #    signal_ar = np.fromfile(fid, np.float32).reshape((-1, 2)).T

    #excluded 41.7e3 
    bw_list = [125e3,250e3,500e3]

    sig = np.load("data.npy")
    sig = sig[0:int((3.2e6))] #signal_ar[0,:]+ 1j * signal_ar[1,:]
    

    time1 = time.time()
    calc_preamble(sig,500e3)
    print(time.time()-time1)
    
    X_shape = (2,len(sig))
    X = RawArray('d', 2 * len(sig))
    X_np = np.frombuffer(X).reshape(X_shape)
    np.copyto(X_np, np.array([sig.real,sig.imag]))

    #with ProcessPoolExecutor(max_workers=5) as executor:
    #    for result in executor.map(calc_preamble, [125e3,250e3,500e3],sig):
    #        print(result)

    time1 = time.time()
    with Pool(processes=len(bw_list) ,initializer=init_worker, initargs=(X, X_shape)) as pool:
        result = pool.map(calc_preamble_multi, bw_list)
        print('Results (pool):\n', np.array(result))

    print(time.time()-time1)
    #for i in bw_list:
    #    process = Thread(target=calc_preamble, args=[sig,7,i,3.2e6,0])
    #    process.start()
    #    threads.append(process)

    #for process in threads:
    #    process.join()
