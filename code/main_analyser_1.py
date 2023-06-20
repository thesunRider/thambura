import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp,hilbert,butter, lfilter, freqz , correlate ,find_peaks, welch,filtfilt
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import DBSCAN,KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_regression import KernelReg
import time,math
import scipy.stats as stats
from matplotlib.patches import Rectangle

from rtlsdr import RtlSdr



max_freq_standard = 400e3 #frequency that is maximum in the spectrum,used for normalisation 400e3 is enough for 1mhz samplerate
threshold_signal = 10e-10 #threshold that determines how  strong the measuring signal should be
confidence_level_boundary = 0.96 # used to set bound limits from centroid,hiher value means larger boundary ,lower value tighter boundary
window_len = 256 # window length used for blackman windowing

def load_data():
	meta_file_path = "res\\usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-meta"
	data_file_path = "res\\usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-data"
	metadata = json.load(open(meta_file_path, "r"))
	meta_global = metadata["global"]
	meta_capture = metadata["captures"][0]

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

	data = np.fromfile(data_file_path,dtype=np.complex64)
	return {"data":data,"sr":sample_rate,"sf":sf,"bw":bw,"prlen":prlen,"crc":crc,"implicit":implicit}


def extract_seperate_peaks(signal_data,sample_rate,peak_height,noise_level,peak_threshold):
	pxx,freq = plt.psd(signal_data,Fs=sample_rate)
	mean_power = np.mean(pxx)
	y_peaks_index = find_peaks(pxx,height=peak_height + noise_level,threshold=peak_threshold)[0]
	px_plot = 10*np.log10(pxx[y_peaks_index]) 


	# compute cumulative distance
	cumulative_distance = np.zeros(pxx.shape[0])
	for i in range(1, pxx.shape[0]):
		cumulative_distance[i] = cumulative_distance[i-1] + np.abs(pxx[i] - pxx[i-1])

	y_sorted = sorted(y_peaks_index)

	cum_threshold = 1e-10 # this threshold defines maximum distance along the graph the peaks should be distant to be considered in the same group 
	cluster_array = []

	for i in range(len(y_peaks_index)-1):
		if  (cumulative_distance[y_peaks_index[i+1]] - cumulative_distance[y_peaks_index[i]]) < cum_threshold:
			cluster_array[-1].append(y_peaks_index[i])
		elif not i == 0 and (cumulative_distance[y_peaks_index[i]] - cumulative_distance[y_peaks_index[i-1]]) < cum_threshold:
			cluster_array[-1].append(y_peaks_index[i])
		else:
			cluster_array.append([])
			cluster_array[-1].append(y_peaks_index[i])
			cluster_array.append([])

	#add last entry
	if (cumulative_distance[y_peaks_index[-1]] - cumulative_distance[y_peaks_index[-2]]) < cum_threshold:
		cluster_array[-1].append(y_peaks_index[-1])
	else:
		cluster_array.append([])
		cluster_array[-1].append(y_peaks_index[-1])
		cluster_array.append([])

	cluster_color = []
	for x in cluster_array:
		cluster_color += [x[0]]*len(x)

	plt.scatter(freq[y_peaks_index], px_plot, c=cluster_color, cmap='Paired')
	plt.plot([-sample_rate/2,sample_rate/2], [ 10*np.log10(mean_power), 10*np.log10(mean_power)], marker = 'o')
	plt.show()



def extract_timebase(signal_data,sample_rate):
	# Define the threshold value
	spectrogram, freqenciesFound, time, imageAxis = plt.specgram(signal_data,Fs=sample_rate,Fc=0,window=np.blackman(window_len))


	# Print the time indices where the signal crosses the threshold
	indices = np.where(spectrogram > threshold_signal)
	print('Signal crossed threshold at time indices:', indices)


	x,y = time[indices[1]],freqenciesFound[indices[0]]/max_freq_standard
	stck = np.hstack((np.array([x]).T,np.array([y]).T))


	# Try different numbers of clusters
	clustering =  DBSCAN(eps=1e-2, min_samples=5)
	y_pred = clustering.fit_predict(stck)
	print("no labels:",len(np.unique(clustering.labels_)),"max freq:",max_freq_standard)

	# Plot the data points with different colors for different clusters
	plt.scatter(stck[:, 0], stck[:, 1]*max_freq_standard, c=y_pred, cmap='Paired')
	plt.title('DBSCAN Clustering')
	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	return clustering,stck

def calculate_cluster_info(specgram,cluster):
	labels = cluster.labels_
	cluster_data = []
	for cluster_label in np.unique(labels):
		if cluster_label == -1:
			continue
		
		cluster_points = specgram[labels == cluster_label]
		cluster_centroid = np.mean(cluster_points, axis=0) * np.array([1,max_freq_standard]) 

		std_dev = np.std(cluster_points,axis=0)
		# Calculate the z-score corresponding to the confidence level
		z_score = stats.norm.ppf(confidence_level_boundary)

		# Calculate the estimated maximum value
		max_value = z_score * std_dev * np.array([1,max_freq_standard])
		cluster_data.append({"label":cluster_label,"centroid":cluster_centroid,"bound_values":max_value})
		bandwidth_cluster = max_value[1] *2

		center = cluster_centroid
		width = max_value[0]*2
		height = max_value[1]*2

		# Calculate the position of the lower left corner of the rectangle
		x = center[0] - width/2
		y = center[1] - height/2

		rect = Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')

		# Add the rectangle patch to the axes
		ax = plt.gca()
		ax.add_patch(rect)

		plt.scatter(cluster_centroid[0],cluster_centroid[1])
		print(f"Cluster {cluster_label}: {cluster_centroid} , bw : {bandwidth_cluster}")

	plt.show()
	return cluster_data

def extract_signal_from_cluster(signal_data,cluster_info,sample_rate):
	#{"label":cluster_label,"centroid":cluster_centroid,"bound_values":max_value}
	bw = cluster_info["bound_values"][1]*2
	centre_frequency = cluster_info["centroid"][1]

	time_stamp_start = cluster_info["centroid"][0] - cluster_info["bound_values"][0]
	time_stamp_end = cluster_info["centroid"][0] + cluster_info["bound_values"][0]

	signal_extracted = signal_data[int(time_stamp_start*sample_rate)-int(sample_rate/1000):int(time_stamp_end*sample_rate)+int(sample_rate/1000)]

	#shift signal to zero frequency
	t = np.linspace(0, len(signal_extracted)/sample_rate, len(signal_extracted))  # time vector 

	shift_f = centre_frequency  # shift by -10e5 Hz
	shift = np.exp(-2j*np.pi*shift_f*t).astype(np.complex64)

	shifted_signal = shift* signal_extracted

	# Define the downsampling factor and low pass filter parameters
	downsample_factor = math.ceil(sample_rate/bw)
	cutoff_frequency = bw/2  # Hz

	b, a = butter(6, cutoff_frequency, 'low',fs=sample_rate)

	# Apply the low pass filter
	x_filtered = filtfilt(b, a, shifted_signal)

	# Downsample the filtered signal
	x_downsampled = x_filtered[::downsample_factor]

	print(f"Cut off : {cutoff_frequency} , new_fs : {sample_rate/downsample_factor}")
	return {"signal":x_downsampled,"fs":sample_rate/downsample_factor,"bw":bw,"fc":centre_frequency,"tia":cluster_info["bound_values"][0] *2}

def analyse_noise_level(signal,sample_rate):
	return np.mean(plt.psd(signal,Fs=sample_rate))

#data = load_data()

#signal = data["data"]
sample_rate = 3.2e6 #data["sr"]
duration_scan = 20 # in seconds

#sdr = RtlSdr()

# some defaults
#sdr.rs = sample_rate
#sdr.fc = 433e6
#sdr.gain = 10

#signal = sdr.read_samples(sample_rate * duration_scan)
#print(signal)

#sdr.close()
noise = np.load('noise.npy') 
signal = np.load('data.npy') 
t = np.linspace(0, len(signal)/sample_rate, len(signal))  # time vector 


#creating a signal workspace
#shift_f = 300e3  # shift by -10e5 Hz
#shift = np.exp(-2j*np.pi*shift_f*t).astype(np.complex64)

#delay_sec = 0.5
#delay_samples = int(delay_sec * sample_rate)

#shifted_signal = np.roll(shift*signal,delay_samples)
#signal_shifted = signal + shifted_signal

noise_level = analyse_noise_level(noise,sample_rate)


## start of processing
start = time.time()
extract_seperate_peaks(signal ,sample_rate,noise_level)
end = print("elapsed1:",time.time() - start)

exit()

start = time.time()
clusters,specgram = extract_timebase(signal ,sample_rate)
end = print("elapsed2:",time.time() - start)

start = time.time()
cluster_info = calculate_cluster_info(specgram,clusters)
end = print("elapsed3:",time.time() - start)

start = time.time()
extracted_singal = extract_signal_from_cluster(signal,cluster_info[0],sample_rate)
end = print("elapsed3:",time.time() - start)