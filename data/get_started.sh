#!usr/bin/env bash

echo 'Downloading the UTKFaces dataset.....'
./gdown.pl https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view UTKFace.tar.gz

echo 'Extracting the UTKFaces dataset.....'
tar -xzf UTKFace.tar.gz

echo 'Correcting mislabelled datapoints...'
mv ./UTKFace/39_1_20170116174525125.jpg.chip.jpg ./UTKFace/39_1_1_20170116174525125.jpg.chip.jpg
mv ./UTKFace/61_1_20170109150557335.jpg.chip.jpg ./UTKFace/61_1_3_20170109150557335.jpg.chip.jpg
mv ./UTKFace/61_1_20170109142408075.jpg.chip.jpg ./UTKFace/61_1_1_20170109142408075.jpg.chip.jpg

echo 'Starting pre-processing....'
python data_analysis.py