#!/usr/bin/env bash

selected_datasets=("acsincome" "airlines" "codrna" "covtype" "creditcard" "diamonds" "higgs" "sensitvehicle")

n="20000"
d="1000"
top_eigvals="20000"

for selected_dataset in "${selected_datasets[@]}"; do
	python ../run_test_spectrum.py 0 --selected_datasets "$selected_dataset" --n $n --d $d --implicit_kernel False --lamb $(echo "$n * 1e-8" | bc -l) --top_eigvals $top_eigvals
	python ../plot_run_test_spectrum.py --selected_dataset $selected_dataset
done