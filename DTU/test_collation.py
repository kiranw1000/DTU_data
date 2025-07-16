import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os, math, tqdm
from collation import split_eeg

os.chdir("/Users/kiran/Documents_local/ASPIRE/data/DTU")

print(os.getcwd())

expinfo = pd.read_csv("S8.csv")
i = 8
mat = sio.loadmat(os.path.join(".", "EEG", f"S{i}.mat"), squeeze_me=True, struct_as_record=False)
events = pd.read_csv(os.path.join(".", f"S{i}.csv"))
wavfiles = set([x for x in events[["wavfile_male", "wavfile_female"]].values.flatten() if type(x) == str])
wavfiles = [(sio.wavfile.read(os.path.join(".", "AUDIO", x))) for x in wavfiles]
minWavLen = min([x[1].shape[0]/x[0] for x in wavfiles if x[0] is not None])
samples = split_eeg(mat, events, wavMinLength=math.ceil(minWavLen * mat["data"].fsample.eeg))