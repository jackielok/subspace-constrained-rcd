#!/usr/bin/env bash

selected_dataset="lhc"

n_samples="10"
samples_indices=($(seq 0 9))

d_list="0 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000"
d_list_indices=($(seq 0 12))

n="100000"
l="1000"
n_iter_cd="5000"
n_iter_cg="50"

for d_list_idx in "${d_list_indices[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test_varyd.py "$job_idx" "$d_list_idx" --selected_dataset $selected_dataset --n $n --l $l --d_list $d_list --n_iter_cd $n_iter_cd --method cg --n_samples $n_samples
	done
done

python ../plot_run_test_varyd.py --selected_dataset $selected_dataset --l $l --d_list $d_list --n_samples $n_samples