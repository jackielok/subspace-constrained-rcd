#!/usr/bin/env bash

samples_indices=($(seq 0 1))
d_list_indices=($(seq 0 2))

d_list="0 50 100"
selected_dataset="lhc"
n_samples="2"

n="1000"
l="100"
n_iter_cd="2000"
n_iter_cg="200"

for d_list_idx in "${d_list_indices[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test_varyd.py "$job_idx" "$d_list_idx" --selected_dataset $selected_dataset --n $n --l $l --d_list $d_list --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --method direct --n_samples $n_samples
	done
done

python ../plot_run_test_varyd.py --selected_dataset $selected_dataset --d_list $d_list --l $l --n_samples $n_samples