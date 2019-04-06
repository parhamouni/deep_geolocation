#!/usr/bin/env bash
#python -m graphsage.supervised_train --train_prefix ./cmu_graphsage_hamilton/cmu --model graphsage_mean ;
#python -m graphsage.supervised_train --train_prefix ./cmu_graphsage_hamilton/cmu --model graphsage_seq
# python -m graphsage.supervised_train --train_prefix ./cmu_graphsage_hamilton/cmu --model graphsage_meanpool ;
for dataset in cmu na world
do
    if [[ "$dataset" = "cmu" ]]
    then
        cel=10
        bucket=50
    elif [[ "$dataset" = "na" ]]
    then
        bucket=2400
        cel=15
    else
        bucket=2400
        cel=5
    fi
    echo '******************** dataset ***************************'
    echo $dataset
    for model in graphsage_meanpool graphsage_seq graphsage_mean gcn
    do
    echo '******************** model type ***************************'
    echo $model
        for dropout in 0.6 0.70 0.75
        do
        echo '******************** dropout ***************************'
        echo $dropout
        for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
            do
            echo '******************** fraction ***************************'
            echo $fraction
            python  graphsage_dataprep.py --bucket $bucket -mindf 10 -cel $cel  -d datasets/$dataset -lblfraction $fraction -seed 77
            done
        done
     done
done
exit 0
