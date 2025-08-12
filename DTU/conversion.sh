#!/usr/bin/env bash
# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate aspire

#####
# Modify these lines
data_dir=$(pwd)
output_dir=$(pwd)/DTU  # Directory to save the converted data
sample_spacing=1
sample_length=10
val_subjects=2
test_subjects=2
low_cutoff=None  # Low cutoff frequency for filtering (Hz)
high_cutoff=None  # High cutoff frequency for filtering (Hz)
resample_freq=128  # Resampling frequency (Hz), if None, no resampling is done
sample_length_mean=5.5
sample_length_std=2.6
max_length=10
min_length=1
#####

# Populate optional args
args=""
if [ "$low_cutoff" != "None" ]; then
    args="$args --low_cutoff $low_cutoff"
fi
if [ "$high_cutoff" != "None" ]; then
    args="$args --high_cutoff $high_cutoff"
fi
if [ "$sample_spacing" != "None" ]; then
    args="$args --sample_spacing $sample_spacing"
fi
if [ "$sample_length" != "None" ]; then
    args="$args --sample_length $sample_length"
fi
if [ "$val_subjects" != "None" ]; then
    args="$args --val_subjects $val_subjects"
fi
if [ "$test_subjects" != "None" ]; then
    args="$args --test_subjects $test_subjects"
fi
if [ "$resample_freq" != "None" ]; then
    args="$args --resample_freq $resample_freq"
fi
if [ "$sample_length_mean" != "None" ]; then
    args="$args --sample_length_mean $sample_length_mean"
fi
if [ "$sample_length_std" != "None" ]; then
    args="$args --sample_length_std $sample_length_std"
fi
if [ "$max_length" != "None" ]; then
    args="$args --max_length $max_length"
fi
if [ "$min_length" != "None" ]; then
    args="$args --min_length $min_length"
fi

# Run the conversion script
python conversion.py --data_dir "$data_dir" --output_dir "$output_dir" $args