#!/usr/bin/env bash

selected_dataset="simlowrank400"

n_samples="100"
samples_indices=($(seq 0 99))

n="8192"
r="400"
l="500"
d="500"
n_iter_cd="5000"
n_iter_cg="300"

python ../run_test_psd.py 0 --selected_dataset $selected_dataset --n $n --r $r --d 0 --l $l --method direct --n_iter_cd 0 --n_iter_cg 0 --n_samples 1 --simulate_dataset

for job_idx in "${samples_indices[@]}"; do
	python ../run_test_psd.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --method direct --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
done

python ../plot_run_test_psd.py --selected_dataset $selected_dataset --n_samples $n_samples