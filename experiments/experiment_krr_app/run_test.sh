#!/usr/bin/env bash

selected_datasets=("acsincome" "airlines" "codrna" "covtype" "creditcard" "diamonds" "higgs" "sensitvehicle")

n_samples="10"
samples_indices=($(seq 0 9))

n="20000"
l="1000"
d="1000"
n_iter_cd="6000"
n_iter_cg="300"

for selected_dataset in "${selected_datasets[@]}"; do
	for job_idx in "${samples_indices[@]}"; do
		python ../run_test.py "$job_idx" --selected_dataset $selected_dataset --n $n --d $d --l $l --implicit_kernel True --lamb $(echo "$n * 1e-8" | bc -l) --method cg --n_iter_cd $n_iter_cd --n_iter_cg $n_iter_cg --n_samples $n_samples
	done
done

python ../plot_run_test.py --selected_dataset acsincome --n_samples $n_samples --legend_loc "lower left"
python ../plot_run_test.py --selected_dataset airlines --n_samples $n_samples --legend_loc "lower left"
python ../plot_run_test.py --selected_dataset codrna --n_samples $n_samples --legend_loc "(0.65, 0.25)"
python ../plot_run_test.py --selected_dataset diamonds --n_samples $n_samples --legend_loc "(0.65, 0.25)"
python ../plot_run_test.py --selected_dataset covtype --n_samples $n_samples --legend_loc "lower left"
python ../plot_run_test.py --selected_dataset creditcard --n_samples $n_samples --legend_loc "lower left"
python ../plot_run_test.py --selected_dataset higgs --n_samples $n_samples --legend_loc "lower left"
python ../plot_run_test.py --selected_dataset sensitvehicle --n_samples $n_samples --legend_loc "lower left"