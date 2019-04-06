#!/usr/bin/env bash
#!/usr/bin/env bash
echo '********************* Download raw dataset ***************************'
echo '*********************Files should be saved to /datasets**********************'
echo '*********************running for models for all datasets ***********************'
for datasetname in na world
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

	for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6 1
	do
        for dim in 100 128 200 300
        do
            for negative in 1 2 5 10
            do
                for windowsize in 5 10 20 30
                do
                    if [[ $fraction = 0.01 ]]
                    then
                    echo ''
                        python -m memory_profiler NetMF/predict_parham.py -bucket $bucket -mindf 10 -cel $cel  -d datasets/$datasetname  -bmf -lblfraction $fraction -seed 77 --window $windowsize --negative $negative --dim $dim
                        python -m memory_profiler NetMF/predict_parham.py -bucket $bucket -mindf 10 -cel $cel  -d datasets/$datasetname  -bmf -lblfraction $fraction -seed 77 --window $windowsize --doc2vec --negative $negative --dim $dim
                    else
                        python -m memory_profiler NetMF/predict_parham.py -bucket $bucket -mindf 10 -cel $cel  -d datasets/$datasetname  -lblfraction $fraction -seed 77 --window $windowsize --negative $negative --dim $dim
                        python -m memory_profiler NetMF/predict_parham.py -bucket $bucket -mindf 10 -cel $cel  -d datasets/$datasetname  -lblfraction $fraction -seed 77 --doc2vec --window $windowsize --negative $negative --dim $dim
                    fi
                done
            done
        done
    done
done
exit
## TODO threading options
## TODO futures is the module that I should go through
## change popen and search see how it goes

## fluent python