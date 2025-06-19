#!/usr/bin/env bash

job_indices=($(seq 0 0))

selected_datasets="simlowrank50"
selected_datasets_arr=($selected_datasets)

n="1024"
d="100"
top_eigvals="1023"

for job_idx in "${job_indices[@]}"; do
	python ../run_test_psd_spectrum.py $job_idx --selected_datasets $selected_datasets --n $n --d $d --top_eigvals $top_eigvals
	python ../plot_run_test_psd_spectrum.py --selected_dataset ${selected_datasets_arr[$job_idx]}
done