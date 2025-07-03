#!/usr/bin/env bash

selected_dataset="simlowrank400"

n_samples="100"
samples_indices=($(seq 0 99))

l_list="10 100 200 300 400 500 600 700 800 900 1000"
l_list_indices=($(seq 0 10))

n="8192"
d="500"
n_epochs="200"

for l_list_idx in "${l_list_indices[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test_psd_varyl.py "$job_idx" "$l_list_idx" --selected_dataset $selected_dataset --n $n --d $d --l_list $l_list --n_epochs $n_epochs --method direct --n_samples $n_samples
	done
done

python ../plot_run_test_psd_varyl.py --selected_dataset $selected_dataset --l_list $l_list --d $d --n_samples $n_samples