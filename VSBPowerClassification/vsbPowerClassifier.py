# Source: https://datamadness.github.io/time-signal-CNN

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pyarrow.parquet as pq
from scipy import signal
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import math


# Execute STFT on phase signal data and reduce the resulting 2D matrix
def signal_stft(phase_data,plot = False):
    fs = 40e6
    f, t, Zxx = signal.stft(phase_data, fs, nperseg=1999, boundry=None)

    reducedZ = block_reduce(np.abs(Zxx), block_size=(1, 1), func=np.max)

    reducedf = f[0::1]
    reducedt = t[0::1]

    if plot:
        plt.pcolormesh(reducedt, reducedf, reducedZ, vmin=0, vmax=0.5)
        plt.title('STFT Magnitude Reduced')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [ms]')
        plt.show()
    return reducedZ
