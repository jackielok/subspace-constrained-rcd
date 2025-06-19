#!/usr/bin/env bash

job_indices=($(seq 0 0))

selected_datasets="sensorless"
selected_datasets_arr=($selected_datasets)

n="20000"
d="1000"
top_eigvals="19999"

for job_idx in "${job_indices[@]}"; do
	python ../run_test_spectrum.py $job_idx --selected_datasets $selected_datasets --n $n --d $d --implicit_kernel False --lamb $(echo "$n * 1e-7" | bc -l) --top_eigvals $top_eigvals
	python ../plot_run_test_spectrum.py --selected_dataset ${selected_datasets_arr[$job_idx]}
done