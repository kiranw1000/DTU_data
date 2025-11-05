import argparse
import os
import shutil
import pandas as pd
from collections import defaultdict
import tqdm
import numpy as np

def main(args):
    if args.mode == "subject_invariance":
        subject_invariance(args)
    elif args.mode == "interface_invariance":
        interface_invariance(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    
def interface_invariance(args):
    mix = pd.read_csv(args.input_csv)
    mix.columns = ["split", "subject", "trial", "tgt_audio", "tgt_start", "", "int_audio", "int_start", "snr", "length"]
    trial_to_audio_pairs = {tuple([x[0],x[1]]): [x[2],x[3]] for x in set(mix[["subject", "trial", "tgt_audio","int_audio", "split"]].itertuples(index=False)) if x[4]=="train"}
    audio_pairs_to_trials = defaultdict(set)
    for k, v in trial_to_audio_pairs.items():
        audio_pairs_to_trials[tuple(v)].add(k)
    subj_int_pairs_to_trials = defaultdict(set)
    for k, v in trial_to_audio_pairs.items():
        subj_int_pairs_to_trials[(k[0], v[1])].add(k)
    subj_attn_pairs_to_trials = defaultdict(set)
    for k, v in trial_to_audio_pairs.items():
        subj_attn_pairs_to_trials[(k[0], v[0])].add(k)
    output = []
    length_distribution = np.random.default_rng(seed=args.seed) 
    spacing_distribution = np.random.default_rng(seed=args.seed) 
    for k, v in tqdm.tqdm(trial_to_audio_pairs.items(), desc="Generating contrastive samples"):
        # Positive samples: same subject and attended, different interference
        time_to_sample = int(args.min_trial_length - args.max_length)
        j = 0
        subbar = tqdm.tqdm(total=time_to_sample, desc=f"Sampling positive pairs for {k}", leave=False)
        while j < time_to_sample:
            temp = length_distribution.normal(args.sample_length_mean, args.sample_length_std)
            temp = max(min(temp, args.max_length), args.min_length)
            sample_length = temp if j+temp < time_to_sample else time_to_sample - j
            for subject, trial in subj_attn_pairs_to_trials[(k[0], v[0])].difference({k}):
                output.append(["",k[0], k[1], v[0], j, "", trial_to_audio_pairs[subject, trial][1], j, subject, trial, trial_to_audio_pairs[subject, trial][0], j, 1,0, sample_length])
            inc = max(spacing_distribution.normal(args.spacing_mean, args.spacing_std), .1)
            j += inc
            subbar.update(inc)
        subbar.close()
        # Negative samples: same subject and interference, different attended
        j = 0
        subbar = tqdm.tqdm(total=time_to_sample, desc=f"Sampling negative pairs for {k}", leave=False)
        while j < time_to_sample:
            temp = length_distribution.normal(args.sample_length_mean, args.sample_length_std)
            temp = max(min(temp, args.max_length), args.min_length)
            sample_length = temp if j+temp < time_to_sample else time_to_sample - j
            for subject, trial in subj_int_pairs_to_trials[(k[0], v[1])].difference({k}):
                output.append(["",k[0], k[1], v[0], j, "", v[1], j, subject, trial, trial_to_audio_pairs[subject, trial][0], j, 0, 0, sample_length])
            inc = max(spacing_distribution.normal(args.spacing_mean, args.spacing_std), .1)
            j += inc
            subbar.update(inc)
        subbar.close()
    output = pd.DataFrame(output, columns=["split", "subject_1", "trial_1", "tgt_audio_1", "tgt_start_1", "", "int_audio", "int_start", "subject_2", "trial_2", "tgt_audio_2", "tgt_start_2", "type", "snr", "length"])
    if args.randomized:
        output = output.sample(frac=1).reset_index(drop=True)
    test_count = int(args.test_split * output.shape[0])
    val_count = int(args.val_split * output.shape[0])
    train_count = output.shape[0] - val_count - test_count
    output.loc[:train_count, "split"] = "train"
    output.loc[train_count:train_count+val_count, "split"] = "val"
    output.loc[train_count+val_count:, "split"] = "test"
    output.to_csv(args.output_csv, index=False)
    print(f"Created {len(output)} samples and saved to {args.output_csv}")

def subject_invariance(args):
    mix = pd.read_csv(args.input_csv)
    mix.columns = ["split", "subject", "trial", "tgt_audio", "tgt_start", "", "int_audio", "int_start", "snr", "length"]
    trial_to_audio_pairs = {tuple([x[0],x[1]]): [x[2],x[3]] for x in set(mix[["subject", "trial", "tgt_audio","int_audio", "split"]].itertuples(index=False)) if x[4]=="train"}
    audio_pairs_to_trials = defaultdict(set)
    for k, v in trial_to_audio_pairs.items():
        audio_pairs_to_trials[tuple(v)].add(k)
    subj_int_pairs_to_trials = defaultdict(set)
    for k, v in trial_to_audio_pairs.items():
        subj_int_pairs_to_trials[(k[0], v[1])].add(k)
    output = []
    length_distribution = np.random.default_rng(seed=args.seed) 
    spacing_distribution = np.random.default_rng(seed=args.seed) 
    for k, v in tqdm.tqdm(trial_to_audio_pairs.items(), desc="Generating contrastive samples"):
        # Positive samples: same interference and attended, different subject
        time_to_sample = int(args.min_trial_length - args.max_length)
        j = 0
        subbar = tqdm.tqdm(total=time_to_sample, desc=f"Sampling positive pairs for {k}", leave=False)
        while j < time_to_sample:
            temp = length_distribution.normal(args.sample_length_mean, args.sample_length_std)
            temp = max(min(temp, args.max_length), args.min_length)
            sample_length = temp if j+temp < time_to_sample else time_to_sample - j
            for subject, trial in audio_pairs_to_trials[tuple(v)].difference({k}):
                output.append(["",k[0], k[1], v[0], j, "", v[1], j, subject, trial, trial_to_audio_pairs[subject, trial][0], j, 1, 0, sample_length])
            inc = max(spacing_distribution.normal(args.spacing_mean, args.spacing_std), .1)
            j += inc
            subbar.update(inc)
        subbar.close()
        # Negative samples: same interference and subject, different attended
        j = 0
        subbar = tqdm.tqdm(total=time_to_sample, desc=f"Sampling negative pairs for {k}", leave=False)
        while j < time_to_sample:
            temp = length_distribution.normal(args.sample_length_mean, args.sample_length_std)
            temp = max(min(temp, args.max_length), args.min_length)
            sample_length = temp if j+temp < time_to_sample else time_to_sample - j
            for subject, trial in subj_int_pairs_to_trials[(k[0], v[1])].difference({k}):
                output.append(["",k[0], k[1], v[0], j, "", v[1], j, subject, trial, trial_to_audio_pairs[subject, trial][0], j, 0, 0, sample_length])
            inc = max(spacing_distribution.normal(args.spacing_mean, args.spacing_std), .1)
            j += inc
            subbar.update(inc)
        subbar.close()
    output = pd.DataFrame(output, columns=["split", "subject_1", "trial_1", "tgt_audio_1", "tgt_start_1", "", "int_audio_1", "int_start_1", "int_audio_2", "int_start_2", "tgt_audio_2", "tgt_start_2", "type", "snr", "length"])
    if args.randomized:
        output = output.sample(frac=1).reset_index(drop=True)
    test_count = int(args.test_split * output.shape[0])
    val_count = int(args.val_split * output.shape[0])
    train_count = output.shape[0] - val_count - test_count
    output.loc[:train_count, "split"] = "train"
    output.loc[train_count:train_count+val_count, "split"] = "val"
    output.loc[train_count+val_count:, "split"] = "test"
    output.to_csv(args.output_csv, index=False)
    print(f"Created {len(output)} samples and saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DTU Mix List Generation for Contrastive Learning")
    io = parser.add_argument_group("IO")
    io.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    io.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    hyperparams = parser.add_argument_group("Hyperparameters")
    hyperparams.add_argument("--min_trial_length", type=float, default=50, help="Minimum length of eeg trials in seconds")
    hyperparams.add_argument("--max_length", type=float, default=4, help="Maximum length of individual audio segments in seconds")
    hyperparams.add_argument("--min_length", type=float, default=1, help="Minimum length of individual audio segments in seconds")
    hyperparams.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    hyperparams.add_argument("--sample_length_mean", type=float, default=3, help="Mean of the normal distribution to sample audio segment lengths from")
    hyperparams.add_argument("--sample_length_std", type=float, default=1, help="Standard deviation of the normal distribution to sample audio segment lengths from")
    hyperparams.add_argument("--spacing_mean", type=float, default=1, help="Mean of the normal distribution to sample spacing between audio segments from")
    hyperparams.add_argument("--spacing_std", type=float, default=0.5, help="Standard deviation of the normal distribution to sample spacing between audio segments from")
    hyperparams.add_argument("--val_split", type=float, default=.1, help="Number of subjects for validation")
    hyperparams.add_argument("--test_split", type=float, default=.15, help="Number of subjects for testing")
    parser.add_argument("--randomized", type=bool, default=True, help="Whether to randomize the order of the output samples")
    parser.add_argument("--mode", type=str, choices=["subject_invariance", "interface_invariance"], default="subject_invariance", help="Mode of contrastive sample generation")
    args = parser.parse_args()
    main(args)