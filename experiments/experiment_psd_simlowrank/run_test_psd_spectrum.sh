#!/usr/bin/env bash

selected_datasets="simlowrank400"
selected_datasets_arr=($selected_datasets)

job_indices=($(seq 0 0))

n="8192"
d="500"
top_eigvals="8192"

for job_idx in "${job_indices[@]}"; do
	python ../run_test_psd_spectrum.py $job_idx --selected_datasets $selected_datasets --n $n --d $d --top_eigvals $top_eigvals
	python ../plot_run_test_psd_spectrum.py --selected_dataset ${selected_datasets_arr[$job_idx]}
done