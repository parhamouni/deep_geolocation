#!/usr/bin/env bash

echo '********************* Download raw dataset ***************************'
echo '*********************Files should be saved to /datasets**********************'
echo '*********************running for models for all datasets ***********************'
for datasetname in cmu
do
    if [[ "$datasetname" = "cmu" ]]
    then
        cel=10
        bucket=50
    elif [[ "$datasetname" = "na" ]]
    then
        bucket=2400
        cel=15
    else
        bucket=2400
        cel=5
    fi

	for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
	do
        for p in 1 2
        do
            for q in 1.1 1.2 1.3
            do
            echo '********************* dataset name:' "$datasetname" " p:""$p" "and q:""$q" "*********************"
            python -u node2vec_implementation.py -bucket $bucket -mindf 10 -cel $cel  -d datasets/$datasetname  -lblfraction $fraction --p $p --q $q -seed 77
            echo '********************* same dataset with doc2vec embedding *********************'
            python -u node2vec_implementation.py -bucket $bucket -cel $cel  -d datasets/$datasetname -lblfraction $fraction --p $p --q $q --doc2vec -seed 77

            done
        done
    done
done