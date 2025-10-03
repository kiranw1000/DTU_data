import argparse, os, shutil, tqdm, random, math, mne
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
    print("Sample Length Mean:", args.sample_length_mean)
    print("Sample Length Std:", args.sample_length_std)
    print("Validation Split:", args.val_split)
    print("Test Split:", args.test_split)
    print("Rereference Method:", args.rereference)
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
    subjects = sorted(subjects)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "eeg"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)

    if args.generate_audio:
        if args.resample_audio:
            os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)
            for file in tqdm.tqdm(os.listdir("AUDIO"), desc="Resampling audio files", position=0, leave=True):
                if file.endswith(".wav"):
                    wav, sr = sf.read(os.path.join("AUDIO", file))
                    if sr != args.resample_audio:
                        wav_resampled = signal.resample(wav, int(args.resample_audio * wav.shape[0] / sr))
                        sf.write(os.path.join(args.output_dir, "audio", file), wav_resampled, args.resample_audio)
                    else:
                        shutil.copy2(os.path.join("AUDIO", file), os.path.join(args.output_dir, "audio", file))
            print("Resampled audio files and saved to output directory.")
        else:
            shutil.copytree("AUDIO", os.path.join(args.output_dir, "audio"), dirs_exist_ok=True)
        print("Copied AUDIO directory to output directory.")

    filtering =  args.high_cutoff is not None or args.low_cutoff is not None
    
    resampling = args.resample_freq is not None

    if args.generate_mix:
        output = pd.DataFrame(columns=["split", "subject", "trial", "tgt_audio", "tgt_start", "", "int_audio", "int_start", "snr", "length"])
        trials = []
        # Process each subject
        length_distribution = np.random.default_rng(seed=args.seed)  # For reproducibility
        spacing_distribution = np.random.default_rng(seed=args.seed + 1)  # Different seed for spacing
        for i, subject in tqdm.tqdm(enumerate(subjects[:num_subjects]), total=num_subjects, desc="Processing subjects", position=0, leave=True):
            # Load experiment info and EEG data
            expinfo = pd.read_csv(os.path.join(args.data_dir, 'EEG', f"{subject}.csv"))
            mat = sio.loadmat(os.path.join(args.data_dir, "EEG", f"{subject}.mat"), squeeze_me=True, struct_as_record=False)
            channels = mat['data'].dim.chan.eeg
            sfreq = mat['data'].fsample.eeg
            ch_types = ['eeg'] * 66 + ['eog'] * 6 + ['misc']
            info = mne.create_info(ch_names=channels.tolist(), sfreq=sfreq, ch_types=ch_types)
            raw = mne.io.RawArray(mat['data'].eeg.T, info)
            if args.generate_eeg:
                # Rereference EEG data if specified
                if args.rereference:
                    raw = raw.set_eeg_reference(args.rereference)
                output_eeg_fs = args.resample_freq if resampling else sfreq
                # Filter and resample EEG data if specified
                if filtering:
                    if i == 0:
                        print(f"Filtering EEG data with filter: {args.filter_type} low:{args.low_cutoff} high:{args.high_cutoff}")
                    raw = raw.filter(l_freq=args.low_cutoff, h_freq=args.high_cutoff, method=args.filter_type)
                if resampling:
                    if i == 0:
                        print(f"Resampling EEG data to {args.resample_freq} Hz")
                    raw = raw.resample(args.resample_freq)
            # Split EEG data into trials
            wav_files = set([x for x in expinfo[["wavfile_male", "wavfile_female"]].values.flatten() if type(x) == str])
            wav_files = [(sf.read(os.path.join(args.data_dir, "AUDIO", x))) for x in wav_files]
            wavMinLength = min([x[0].shape[0] / x[1] for x in wav_files if x[0] is not None])
            eeg_data, indices = split_eeg(raw.get_data().T, mat['data'].event.eeg.sample, mat['data'].event.eeg.value, expinfo, wavMinLength=wavMinLength)
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
                time_to_sample = int((eeg_data[trial].shape[0]/output_eeg_fs)  - args.max_length)
                j = 0
                while time_to_sample - j > .1:
                    temp = length_distribution.normal(args.sample_length_mean, args.sample_length_std)
                    temp = max(min(temp, args.max_length), args.min_length)
                    sample_length = temp if j+temp < time_to_sample else time_to_sample - j
                    output = pd.concat([output, pd.DataFrame([["",i+1, trial+1, attn_wav, j, "", int_wav, j, 0, sample_length]], columns=output.columns)])
                    j += max(spacing_distribution.normal(args.spacing_mean, args.spacing_std), .1)
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

        if args.randomized:
            output = output.sample(frac=1).reset_index(drop=True)

        output.to_csv(os.path.join(args.output_dir, "mix.csv"), index=False, header=False)
        print(f"Created {len(output)} samples and saved to {os.path.join(args.output_dir, 'mix.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTU Data Conversion")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing DTU data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save converted data")
    parser.add_argument("--sample_length_mean", type=float, default=5.5, help="Mean length of each EEG sample in seconds")
    parser.add_argument("--sample_length_std", type=float, default=0, help="Standard deviation of each EEG sample length in seconds")
    parser.add_argument("--spacing_mean", type=float, default=.11, help="Mean spacing between audio samples in seconds")
    parser.add_argument("--spacing_std", type=float, default=0, help="Standard deviation of spacing between audio samples in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_length", type=float, default=10, help="Maximum length of audio samples in seconds")
    parser.add_argument("--min_length", type=float, default=1, help="Minimum length of audio samples in seconds")
    parser.add_argument("--val_split", type=float, default=.1, help="Number of subjects for validation")
    parser.add_argument("--test_split", type=float, default=.15, help="Number of subjects for testing")
    parser.add_argument("--high_cutoff", type=float, default=None, help="High cutoff frequency for filtering")
    parser.add_argument("--low_cutoff", type=float, default=None, help="Low cutoff frequency for filtering")
    parser.add_argument("--resample_freq", type=int, default=None, help="Resampling frequency for EEG data")
    parser.add_argument("--resample_audio", type=int, default=8000, help="Resampling frequency for audio data")
    parser.add_argument("--subjects", type=int, default=None, help="Number of subjects to process")
    parser.add_argument("--randomized", action='store_true', help="Randomize the order of trials")
    parser.add_argument("--generate_mix", default=True, help="Create mix file")
    parser.add_argument("--generate_audio", default=True, help="Process audio files")
    parser.add_argument("--generate_eeg", default=True, help="Process EEG files")
    parser.add_argument("--filter_type", type=str, default="fir", choices=["iir", "fir"], help="Type of filter to use for EEG data")
    parser.add_argument("--rereference", type=str, default="average", help="Re-reference EEG data to specified channel")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)