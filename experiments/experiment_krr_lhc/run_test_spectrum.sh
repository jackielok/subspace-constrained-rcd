#!/usr/bin/env bash

selected_datasets="lhc"
selected_datasets_arr=($selected_datasets)

job_indices=($(seq 0 0))

n="100000"
d="1000"
top_eigvals="20000"

for job_idx in "${job_indices[@]}"; do
	python ../run_test_spectrum.py $job_idx --selected_datasets $selected_datasets --n $n --d $d --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --top_eigvals $top_eigvals
	python ../plot_run_test_spectrum.py --selected_dataset ${selected_datasets_arr[$job_idx]}
done