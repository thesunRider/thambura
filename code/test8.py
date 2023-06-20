import datetime as dt
import numpy as np
import sigmf
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str

# suppose we have an complex timeseries signal
data = np.zeros(1024, dtype=np.complex64)

# write those samples to file in cf32_le
data.tofile('example_cf32.sigmf-data')

# create the metadata
meta = SigMFFile(
    data_file='example_cf32.sigmf-data', # extension is optional
    global_info = {
        SigMFFile.DATATYPE_KEY: get_data_type_str(data),  # in this case, 'cf32_le'
        SigMFFile.SAMPLE_RATE_KEY: 48000,
        SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
        SigMFFile.DESCRIPTION_KEY: 'All zero complex float32 example file.',
        SigMFFile.VERSION_KEY: sigmf.__version__,
    }
)

# create a capture key at time index 0
meta.add_capture(0, metadata={
    SigMFFile.FREQUENCY_KEY: 915000000,
    SigMFFile.DATETIME_KEY: dt.datetime.utcnow().isoformat()+'Z',
})

# add an annotation at sample 100 with length 200 & 10 KHz width
meta.add_annotation(100, 200, metadata = {
    SigMFFile.FLO_KEY: 914995000.0,
    SigMFFile.FHI_KEY: 915005000.0,
    SigMFFile.COMMENT_KEY: 'example annotation',
})

# check for mistakes & write to disk
meta.tofile('example_cf32.sigmf-meta') # extension is optional