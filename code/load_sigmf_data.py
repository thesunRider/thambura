import numpy as np
import json

meta_file_path = "usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-meta"
data_file_path = "usrp-868.1-sf7-cr4-bw125-crc-0.sigmf-data"
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