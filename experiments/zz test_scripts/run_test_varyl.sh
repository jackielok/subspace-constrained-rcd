#!/usr/bin/env bash

samples_indices=($(seq 0 1))
l_list_indices=($(seq 0 2))

l_list="50 100 150"
selected_dataset="lhc"
n_samples="2"

n="1000"
d="100"
n_epochs="100"

for l_list_idx in "${l_list_indices[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test_varyl.py "$job_idx" "$l_list_idx" --selected_dataset $selected_dataset --n $n --d $d --l_list $l_list --n_epochs $n_epochs --implicit_kernel True --lamb $(echo "$n * 1e-9" | bc -l) --method direct --n_samples $n_samples
	done
done

python ../plot_run_test_varyl.py --selected_dataset $selected_dataset --l_list $l_list --d $d --n_samples $n_samples