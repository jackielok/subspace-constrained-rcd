#!/usr/bin/env bash

job_indices=($(seq 0 0))

selected_datasets="lhc"
selected_datasets_arr=($selected_datasets)

n="1000"
d="100"
top_eigvals="800"

for job_idx in "${job_indices[@]}"; do
	python ../run_test_spectrum.py $job_idx --selected_datasets $selected_datasets --n $n --d $d --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --top_eigvals $top_eigvals
	python ../plot_run_test_spectrum.py --selected_dataset ${selected_datasets_arr[$job_idx]}
done