import argparse, os, shutil, tqdm
import scipy.io as sio
import pandas as pd
import numpy as np
import soundfile as sf
from collation import split_eeg

def main(args):
    assert args.data_dir != ".", "Please provide a valid data directory."
    os.chdir(args.data_dir)
    print("Current working directory:", os.getcwd())
    
    assert args.output_dir != ".", "Please provide a valid output directory."

    mat_files = [x for x in os.listdir(os.path.join(args.data_dir, "EEG")) if x.endswith(".mat")]
    num_subjects = len(mat_files)
    print(f"Found {num_subjects} subjects in the EEG directory.")
    subjects = [x.split(".")[0] for x in mat_files]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "EEG"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "AUDIO"), exist_ok=True)
    
    shutil.copytree("AUDIO", os.path.join(args.output_dir, "AUDIO"), dirs_exist_ok=True)
    print("Copied AUDIO directory to output directory.")

    output = []
    split = "train"
    for i, subject in tqdm.tqdm(enumerate(subjects)):
        if i > num_subjects - args.test_subjects:
            split = "test"
        elif i > num_subjects - args.test_subjects - args.val_subjects:
            split = "val"
        expinfo = pd.read_csv(os.path.join(args.data_dir, 'EEG', f"{subject}.csv"))
        mat = sio.loadmat(os.path.join(args.data_dir, "EEG", f"{subject}.mat"), squeeze_me=True, struct_as_record=False)
        wav_files = set([x for x in expinfo[["wavfile_male", "wavfile_female"]].values.flatten() if type(x) == str])
        wav_files = [(sf.read(os.path.join(args.data_dir, "AUDIO", x))) for x in wav_files]
        wavMinLength = min([x[0].shape[0] / x[1] for x in wav_files if x[0] is not None])
        eeg_data, indices = split_eeg(mat, expinfo, wavMinLength=wavMinLength)
        expinfo = expinfo.iloc[indices]
        for trial in range(eeg_data.shape[0]):
            wav_m, wav_f = expinfo.iloc[trial][["wavfile_male", "wavfile_female"]].values
            if expinfo.iloc[trial]["attend_mf"] == 1:
                attn_wav, int_wav = wav_m, wav_f
            else:
                attn_wav, int_wav = wav_f, wav_m
            eeg_path = os.path.join(args.output_dir, "EEG", f"{subject}Tra{trial+1}.npy")
            np.save(eeg_path, eeg_data[trial])
            time_to_sample = int(eeg_data[trial].shape[0] - (mat["data"].fsample.eeg * args.sample_length))
            for j in range(0, time_to_sample, args.sample_spacing * mat["data"].fsample.eeg):
                output.append([split, i + 1, trial + 1, attn_wav, j, 0, int_wav, j, 0])
    np.savetxt(os.path.join(args.output_dir, "mix.csv"), output, fmt="%s", delimiter=",")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTU Data Conversion")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing DTU data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted data")
    parser.add_argument("--sample_spacing", type=int, default=1, help="Sample spacing for EEG data")
    parser.add_argument("--sample_length", type=int, default=10, help="Length of each EEG sample in seconds")
    parser.add_argument("--val_subjects", type=int, default=2, help="Number of subjects for validation")
    parser.add_argument("--test_subjects", type=int, default=2, help="Number of subjects for testing")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)