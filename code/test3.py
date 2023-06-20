import numpy as np
import pdb


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


def main():
	import matplotlib.pyplot as plt
	pll = SimPLL(0.002)
	num_samples = 500
	phi = 3.0
	frequency_offset = -0.2
	ref = []
	out = []
	diff = []
	for i in range(0, num_samples - 1):
		in_sig = np.exp(1j*phi)
		phi += frequency_offset
		pll.step(in_sig)
		ref.append(in_sig)
		out.append(pll.vco)
		diff.append(pll.phase_difference)
	#plt.plot(ref)
	plt.plot(ref)
	plt.plot(out)
	plt.plot(diff)
	plt.show()

main()