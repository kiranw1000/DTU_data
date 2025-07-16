import numpy as np
import scipy.io as sio
import pandas as pd

def split_eeg(mat: sio.matlab.mio5_params.mat_struct, expinfo: pd.DataFrame, wavMinLength: int = 50):
    """Splits the EEG data insto samples based on the attended speaker.

    Args:
        mat (sio.matlab.mio5_params.mat_struct): The EEG data structure.
        expinfo (pd.DataFrame): DataFrame containing experimental information.
    """
    eeg_data = mat['data'].eeg
    events = mat['data'].event.eeg.sample
    evalues = mat['data'].event.eeg.value
    
    events = events[::2]
    evalues = evalues[::2]
    
    samples = []
    indices = []
    for i, val in enumerate(evalues[:-1]):
        if val != 191:
            if type(expinfo.iloc[i]["wavfile_female"]) == str:
                samples.append(eeg_data[events[i]:events[i+1]])
                indices.append(i)
    if evalues[-1] != 191 and type(expinfo.iloc[-1]["wavfile_female"]) == str:
        samples.append(eeg_data[events[-1]:])
        indices.append(len(evalues) - 1)

    cutoff = min([x.shape[0] for x in samples])
    cutoff = int(min(cutoff, wavMinLength * mat["data"].fsample.eeg))

    samples = np.array([sample[:cutoff] for sample in samples])
    
    return samples, indices