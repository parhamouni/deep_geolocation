#!/usr/bin/env bash
pip install torch torchvision
pip install --upgrade torch-scatter
pip install --upgrade torch-sparse
pip install --upgrade torch-cluster
pip install --upgrade torch-spline-conv 
pip install torch-geometric
pip install haversine
pip install gensim

echo '********************* Download raw dataset ***************************'
echo '*********************Files should be saved to /datasets**********************'
echo '********************* running for models for cmu dataset***********************'
echo '********************* GAT (11 121) ***********************'

for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn GAT -d datasets/cmu -save_results -lblfraction $fraction
done

echo '********************* GraphSAGE MEAN (hidden layer 400,400, can be changed if you want) ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn GraphSAGE -d datasets/cmu -save_results -lblfraction $fraction
done 

echo '********************* ARMA  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn ARMA -d datasets/cmu -save_results -lblfraction $fraction
done



echo '********************* running for models for cmu dataset with doc2vec***********************'
echo '********************* GAT (11 121) ***********************'

for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn GAT -d datasets/cmu -save_results --doc2vec -lblfraction $fraction
done

echo '********************* GraphSAGE MEAN (hidden layer 400,400, can be changed if you want) ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn GraphSAGE -d datasets/cmu -save_results --doc2vec -lblfraction $fraction
done

echo '********************* ARMA  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 50 -mindf 10 -reg 0.0 -dropout 0.5 -cel 10   -mn ARMA -d datasets/cmu -save_results --doc2vec -lblfraction $fraction
done

echo '********************* running for models for na dataset***********************'



echo '********************* GAT (hidden layer 11,121, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn GAT -d datasets/na -save_results -lblfraction $fraction
done

echo '********************* GraphSAGE MEAN (hidden layer 400,400, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn GraphSAGE -d datasets/na -save_results -lblfraction $fraction
done

echo '********************* ARMA (16, 16)  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn ARMA -d datasets/na -save_results -lblfraction $fraction
done

echo '********************* running for models for na dataset with doc2vec***********************'

echo '********************* GAT (hidden layer 11,121, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn GAT -d datasets/na -save_results --doc2vec -lblfraction $fraction
done

echo '********************* GraphSAGE MEAN (hidden layer 400,400, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn GraphSAGE -d datasets/na -save_results --doc2vec -lblfraction $fraction
done

echo '********************* ARMA (16, 16)  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 15   -mn ARMA -d datasets/na -save_results --doc2vec -lblfraction $fraction
done



echo '********************* running for models for world dataset***********************'
echo '********************* GAT (hidden layer 11,121, can be changed if you want)***********************'

for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn GAT -d datasets/world -save_results -lblfraction $fraction
done

echo '********************* model GraphSAGE MEAN( hidden layer 400,400, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn GraphSAGE -d datasets/world -save_results -lblfraction $fraction
done

echo '********************* ARMA (16, 16)  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn ARMA -d datasets/world -save_results -lblfraction $fraction
done

echo '********************* running for models for world dataset with doc2vec***********************'

echo '********************* GAT (hidden layer 11,121, can be changed if you want)***********************'

for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 11 121  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn GAT -d datasets/world -save_results --doc2vec -lblfraction $fraction
done

echo '********************* model GraphSAGE MEAN( hidden layer 400,400, can be changed if you want)***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 400 400  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn GraphSAGE -d datasets/world -save_results --doc2vec -lblfraction $fraction
done

echo '********************* ARMA (16, 16)  ***********************'
for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
do
python -u main.py -hid 16 16  -bucket 2400 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5   -mn ARMA -d datasets/world -save_results --doc2vec -lblfraction $fraction
done








