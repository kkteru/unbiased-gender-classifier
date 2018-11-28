#!usr/bin/env bash

echo 'Downloading the trained autoencoder models.....'
../data/gdown.pl https://drive.google.com/file/d/1d7Yfhrklfg-4xS4qzWI-riCX-Bvt0rbR/view ae_race_invariant.pth
../data/gdown.pl https://drive.google.com/file/d/1k-pTpdSaRO5q9v3zjXaPvBWSdgiL4mSC/view ae_vanilla.pth