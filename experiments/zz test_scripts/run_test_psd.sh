#!/usr/bin/env bash

samples_indices=($(seq 0 1))

selected_dataset="simlowrank50"
n_samples="2"

n="1024"
r="50"
l="100"
d="100"
n_iter_cd="2000"
n_iter_cg="200"

python ../run_test_psd.py 0 --selected_dataset $selected_dataset --n $n --r $r --d 0 --l $l --method direct --n_iter_cd 0 --n_iter_cg 0 --n_samples 1 --simulate_dataset

for job_idx in "${samples_indices[@]}"; do
	python ../run_test_psd.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --method direct --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
done

python ../plot_run_test_psd.py --selected_dataset $selected_dataset --n_samples $n_samples