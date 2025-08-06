import argparse, os, shutil, tqdm, random, math
import scipy.io as sio
import pandas as pd
import numpy as np
import soundfile as sf
import scipy.signal as signal
from collation import split_eeg

def main(args):
    # Print configuration
    print("Data Directory:", args.data_dir)
    print("Output Directory:", args.output_dir)
    print("Sample Spacing:", args.sample_spacing)
    print("Sample Length:", args.sample_length)
    print("Validation Split:", args.val_split)
    print("Test Split:", args.test_split)
    print("Low Cutoff Frequency:", args.low_cutoff)
    print("High Cutoff Frequency:", args.high_cutoff)
    print("Resampling Frequency:", args.resample_freq)
    
    # Check and set directories
    assert args.data_dir != ".", "Please provide a valid data directory."
    os.chdir(args.data_dir)
    print("Current working directory:", os.getcwd())
    
    assert args.output_dir != ".", "Please provide a valid output directory."

    mat_files = [x for x in os.listdir(os.path.join(args.data_dir, "EEG")) if x.endswith(".mat")]
    num_subjects = len(mat_files)
    print(f"Found {num_subjects} subjects in the EEG directory.")
    if args.subjects is not None:
        num_subjects = min(num_subjects, args.subjects)
        print(f"Limiting processing to {num_subjects} subjects as per the --subjects argument.")
    subjects = [x.split(".")[0] for x in mat_files]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "eeg"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)
    
    if not args.mix_only:
        shutil.copytree("AUDIO", os.path.join(args.output_dir, "audio"), dirs_exist_ok=True)
        print("Copied AUDIO directory to output directory.")

    filtering =  args.high_cutoff is not None or args.low_cutoff is not None
    
    resampling = args.resample_freq is not None

    output = pd.DataFrame(columns=["split", "subject", "trial", "tgt_audio", "tgt_start", "", "int_audio", "int_start", "snr", "length"])
    trials = []
    # Process each subject
    for i, subject in tqdm.tqdm(enumerate(subjects[:num_subjects]), total=num_subjects, desc="Processing subjects", position=0, leave=True):
        # Load experiment info and EEG data
        expinfo = pd.read_csv(os.path.join(args.data_dir, 'EEG', f"{subject}.csv"))
        mat = sio.loadmat(os.path.join(args.data_dir, "EEG", f"{subject}.mat"), squeeze_me=True, struct_as_record=False)
        wav_files = set([x for x in expinfo[["wavfile_male", "wavfile_female"]].values.flatten() if type(x) == str])
        wav_files = [(sf.read(os.path.join(args.data_dir, "AUDIO", x))) for x in wav_files]
        wavMinLength = min([x[0].shape[0] / x[1] for x in wav_files if x[0] is not None])
        # Split EEG data into trials
        eeg_data, indices = split_eeg(mat, expinfo, wavMinLength=wavMinLength)
        if args.mix_only:
            pass
        else:
            # Filter and resample EEG data if specified
            fs = mat["data"].fsample.eeg
            if filtering:
                low = 0 if args.low_cutoff is None else args.low_cutoff / fs
                high = 1 if args.high_cutoff is None else args.high_cutoff / fs
                if low <= 0:
                    ftype = 'lowpass'
                    cutoffs = [high]
                elif high >= 1:
                    ftype = 'highpass'
                    cutoffs = [low]
                else:
                    ftype = 'bandpass'
                    cutoffs = [low, high]
                if i == 0:
                    print(f"Filtering EEG data with {ftype} filter: {cutoffs}")
                sos = signal.butter(4, cutoffs, btype=ftype, output='sos')
                eeg_data = signal.sosfiltfilt(sos, eeg_data, axis=1)
            if resampling:
                resampled_size = int(args.resample_freq * eeg_data.shape[1] / fs)
                eeg_data = signal.resample(eeg_data, resampled_size, axis=1)
                print(f"EEG data shape after resampling: {eeg_data.shape}")
        expinfo = expinfo.iloc[indices]
        # Populate mix file which specifies audio and EEG data correspondence
        for trial in tqdm.tqdm(range(eeg_data.shape[0]), desc="Processing trials", position=1, leave=False):
            trials.append((i+1, trial + 1))
            wav_m, wav_f = expinfo.iloc[trial][["wavfile_male", "wavfile_female"]].values
            if expinfo.iloc[trial]["attend_mf"] == 1:
                attn_wav, int_wav = wav_m, wav_f
            else:
                attn_wav, int_wav = wav_f, wav_m
            eeg_path = os.path.join(args.output_dir, "eeg", f"{subject}Tra{trial+1}.npy")
            np.save(eeg_path, eeg_data[trial])
            time_to_sample = int((eeg_data[trial].shape[0]/mat["data"].fsample.eeg)  - args.sample_length)
            for j in np.arange(0, time_to_sample, args.sample_spacing):
                output = pd.concat([output, pd.DataFrame([["",i+1, trial+1, attn_wav, j, "", int_wav, j, 0, args.sample_length]], columns=output.columns)])
    num_trials = len(trials)
    val_trials = math.floor(args.val_split * num_trials)
    test_trials = math.floor(args.test_split * num_trials)
    train_trials = num_trials - val_trials - test_trials
    print(f"Total trials: {num_trials}, Train: {train_trials}, Val: {val_trials}, Test: {test_trials}")
    
    trial_dict = {}
    for split, number in tqdm.tqdm({"train": train_trials, "val": val_trials, "test": test_trials}.items(), desc="Assigning splits", position=2, leave=False):
        selected_trials = random.sample(trials, number)
        trials = [x for x in trials if x not in selected_trials]
        trial_dict.update({x: split for x in selected_trials})
    output.reset_index(drop=True, inplace=True)
    for i, row in tqdm.tqdm(output.iterrows(), desc="Assigning splits", position=2, leave=False):
        output.at[i, "split"] = trial_dict[(row["subject"], row["trial"])]

    output.to_csv(os.path.join(args.output_dir, "mix.csv"), index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTU Data Conversion")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing DTU data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted data")
    parser.add_argument("--sample_spacing", type=float, default=1, help="Sample spacing for EEG data")
    parser.add_argument("--sample_length", type=float, default=10, help="Length of each EEG sample in seconds")
    parser.add_argument("--val_split", type=float, default=.1, help="Number of subjects for validation")
    parser.add_argument("--test_split", type=float, default=.15, help="Number of subjects for testing")
    parser.add_argument("--high_cutoff", type=float, default=None, help="High cutoff frequency for filtering")
    parser.add_argument("--low_cutoff", type=float, default=None, help="Low cutoff frequency for filtering")
    parser.add_argument("--resample_freq", type=int, default=None, help="Resampling frequency for EEG data")
    parser.add_argument("--subjects", type=int, default=None, help="Number of subjects to process")
    parser.add_argument("--mix_only", action='store_true', help="Only create mix file without processing audio or EEG data")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)