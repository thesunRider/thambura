import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp,hilbert,butter, lfilter, freqz , correlate ,find_peaks, welch
import json
from matplotlib.ticker import FuncFormatter

meta_file_path = "usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-meta"
data_file_path = "usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-data"
metadata = json.load(open(meta_file_path, "r"))
meta_global = metadata["global"]
meta_capture = metadata["captures"][0]

class SimPLL(object):
	def __init__(self, lf_bandwidth):
		self.phase_out = 0.0
		self.freq_out = 0.0
		self.vco = np.exp(1j*self.phase_out)
		self.phase_difference = 0.0
		self.bw = lf_bandwidth
		self.beta = np.sqrt(lf_bandwidth)

	def set_signal_in(self, signal_in):
		"""Set a iterator as input signal"""
		self.signal_in = signal_in

	def signal_out(self):
		"""Generate the output steps"""
		for sample_in in self.signal_in:
			self.step(sample_in)
			yield self.vco

	def update_phase_estimate(self):
		self.vco = np.exp(1j*self.phase_out)

	def update_phase_difference(self, in_sig):
		self.phase_difference = np.angle(in_sig*np.conj(self.vco))

	def step(self, in_sig):
		# Takes an instantaneous sample of a signal and updates the PLL's inner state
		self.update_phase_difference(in_sig)
		self.freq_out += self.bw * self.phase_difference
		self.phase_out += self.beta * self.phase_difference + self.freq_out
		self.update_phase_estimate()



def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



def generate_sawchirp(fs,bw,sf,N_chirps=1,phi_in=0,invert=False):
	Rs = bw/2**sf

	carrier_freq = bw
	duration = 1/Rs

	print(duration,bw)
	f0 = carrier_freq - bw/2           # Starting frequency (Hz)
	f1 = carrier_freq + bw/2       # Ending frequency (Hz)


	t = np.linspace(0, duration, int(fs*duration))
	single_chirp_count = int(fs*duration)
	phi = 180    # Phase shift for sawtooth-like FFT

	if not invert:
		chirp_signal = chirp(t, f0, duration, f1, method='linear', phi=phi)
	else:
		chirp_signal = chirp(t, f1, duration, f0, method='linear', phi=phi)

	chirp_signal = np.tile(chirp_signal,N_chirps)
	chirp_signal = butter_bandpass_filter(chirp_signal,f0,f1,fs,10)

	t = np.linspace(0, N_chirps*duration, int(fs*N_chirps*duration))

	analytic_signal = hilbert(chirp_signal)

	# Compute the mean of the instantaneous frequency to obtain the frequency offset
	f_offset_est = carrier_freq

	# Compute the complex exponential to remove the frequency offset
	iq_signal_no_offset = analytic_signal * np.exp(-1j*2*np.pi*f_offset_est*t)
	normalised = iq_signal_no_offset / max(abs(iq_signal_no_offset))

	angle_to_element = int(phi_in*single_chirp_count/180)
	phase_change = np.roll(normalised, angle_to_element)
	return phase_change #normalize and return

def find_signal_start(signal_in,sample_rate,bw,sf):
	# template_samplerate = sample_rate
	# template  = generate_sawchirp(sample_rate,bw,sf,10,0)

	# # Calculate the cross-correlation of the signal and template
	# corr = correlate(signal_in, template, mode='full')

	# peaks, _ = find_peaks(corr, height=0.9*np.max(corr))
	# print(peaks)
	# # Find the index of the maximum correlation value
	# start_index = peaks[0]

	# # Calculate the start time of the chirp signal based on the sample rate and start index
	# start_time = start_index / sample_rate

	# # Print the start time of the chirp signal
	# print('Chirp signal starts at:', start_time, 'seconds')
	pass

def preprocess_raw_sample(sample_in,fc,bw,fs):
	f_offset_est= 4.25e9
	# Compute the phase angle of the IQ signal
	print("f_offset_est=",f_offset_est)
	t = np.linspace(0,len(sample_in)/fs,len(sample_in))

	# Compute the complex exponential to remove the frequency offset
	sample_out = sample_in * np.exp(-1j*2*np.pi*f_offset_est*t)
	return sample_out

def filter(sample_in):
	num_taps = 101
	beta = 0.45
	Ts = 6 # Assume sample rate is 1 Hz, so sample period is 1, so *symbol* period is 8
	t = np.arange(num_taps) - (num_taps-1)//2
	h = 1/Ts*np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts) / (1 - (2*beta*t/Ts)**2)

	x_shaped = np.convolve(sample_in, h)
	return x_shaped

#fs = 1e4        # Sampling frequency (Hz)
#bw = 1e3
#sf = 8

# Parse metadata. Not using SigMF library because of extra dependency
sample_rate = meta_global["core:sample_rate"]
capture_freq = meta_capture["core:frequency"]
transmit_freq = meta_capture["lora:frequency"]
sf = meta_capture["lora:sf"]
cr = meta_capture["lora:cr"]
bw = int(meta_capture["lora:bw"])
prlen = meta_capture["lora:prlen"]
crc = meta_capture["lora:crc"]
implicit = meta_capture["lora:implicit"]
Rs = bw/2**sf

#18 -p, 1742 - n
signal_data = np.fromfile(data_file_path,dtype=np.complex64)#[int(0.18*sample_rate):int(0.22*sample_rate)] 


time_base = np.linspace(0,len(signal_data)/sample_rate,len(signal_data))

chirp_signal = generate_sawchirp(sample_rate,bw,sf,39,0,True)
signal_data = preprocess_raw_sample(signal_data,capture_freq,bw,sample_rate)

t = len(signal_data) - len(chirp_signal)
signal_data *= np.pad(chirp_signal, pad_width=(0, t), mode='constant',constant_values=1)

signal_data = filter(signal_data)

find_signal_start(signal_data,sample_rate,bw,sf)

ax = plt.gca()
mkfunc = lambda x, pos: '%1.1fM' % (x * 1e-6) if x >= 1e6 else '%1.1fK' % (x * 1e-3) if x >= 1e3 else '%1.1f' % x
mkformatter = FuncFormatter(mkfunc)
ax.yaxis.set_major_formatter(mkformatter)

plt.psd(signal_data,Fs=sample_rate) # detrend x by subtracting the mean
plt.show()

plt.specgram(signal_data,Fs=sample_rate,Fc=0,window=np.blackman(256))
plt.ylabel('Frequency')
plt.xlabel('Time')
plt.title('FFT of chirp signal')
plt.grid(True, which='both', color='gray', linewidth=0.5)
plt.show()