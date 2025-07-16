#!/usr/bin/env bash
# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate aspire

#####
# Modify these lines
data_dir="."
output_dir="./DTU"  # Directory to save the converted data
sample_spacing=1
sample_length=10
val_subjects=2
test_subjects=2
#####
echo data_dir: $data_dir
echo output_dir: $output_dir
echo sample_spacing: $sample_spacing
echo sample_length: $sample_length
echo val_subjects: $val_subjects
echo test_subjects: $test_subjects

# Run the conversion script
python conversion.py --data_dir "$data_dir" --output_dir "$output_dir" --sample_spacing "$sample_spacing" --sample_length "$sample_length" --val_subjects "$val_subjects" --test_subjects "$test_subjects"