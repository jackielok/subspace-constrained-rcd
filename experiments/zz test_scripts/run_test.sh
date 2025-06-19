#!/usr/bin/env bash

samples_indices=($(seq 0 1))

selected_dataset="lhc"
n_samples="2"

n="1000"
l="100"
d="100"
n_iter_cd="2000"
n_iter_cg="200"

for job_idx in "${samples_indices[@]}"; do
	python ../run_test.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --method direct --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
done

python ../plot_run_test.py --selected_dataset $selected_dataset --n_samples $n_samples