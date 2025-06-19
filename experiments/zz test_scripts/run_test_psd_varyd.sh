#!/usr/bin/env bash

samples_indices=($(seq 0 1))
d_list_indices=($(seq 0 2))

d_list="0 50 100"
selected_dataset="simlowrank50"
n_samples="2"

n="1024"
l1="500"
l2="50"
n_iter_cd1="200"
n_iter_cd2="2000"

for d_list_idx in "${d_list_indices[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test_psd_varyd.py "$job_idx" "$d_list_idx" --selected_dataset $selected_dataset --n $n --l $l1 --d_list $d_list --n_iter_cd $n_iter_cd1 --method direct --n_samples $n_samples
		python ../run_test_psd_varyd.py "$job_idx" "$d_list_idx" --selected_dataset $selected_dataset --n $n --l $l2 --d_list $d_list --n_iter_cd $n_iter_cd2 --method direct --n_samples $n_samples
	done
done

python ../plot_run_test_psd_varyd.py --selected_dataset $selected_dataset --d_list $d_list --n_samples $n_samples