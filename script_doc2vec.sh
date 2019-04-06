#!/usr/bin/env bash

#!/usr/bin/env bash
#!/usr/bin/env bash

echo '********************* Download raw dataset ***************************'
echo '*********************Files should be saved to /datasets**********************'
echo '*********************running for models for all datasets ***********************'
for datasetname in cmu na world
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
        python  NetMF/predict_parham.py -bucket $bucket -mindf 10 -cel $cel -d datasets/$datasetname  -lblfraction $fraction -seed 77 --doc2vec

    done
done