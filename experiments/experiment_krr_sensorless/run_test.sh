#!/usr/bin/env bash

samples_indices=($(seq 0 99))

selected_dataset="sensorless"
n_samples="100"

n="20000"
l="1000"
d="1000"
n_iter_cd="6000"
n_iter_cg="300"

for job_idx in "${samples_indices[@]}"; do
	python ../run_test.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --implicit_kernel False --lamb $(echo "$n * 1e-7" | bc -l) --method cg --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
done

python ../plot_run_test.py --selected_dataset $selected_dataset --n_samples $n_samples