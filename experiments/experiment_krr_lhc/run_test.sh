#!/usr/bin/env bash

selected_dataset="lhc"

n_samples="100"
samples_indices=($(seq 0 99))

n="100000"
l="1000"
d="1000"
n_iter_cd="10000"
n_iter_cg="100"

for job_idx in "${samples_indices[@]}"; do
	python ../run_test.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --method cg --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
done

python ../plot_run_test.py --selected_dataset $selected_dataset --n_samples $n_samples --legend_loc "lower left"