# Reducing bias in classifiers by disentangling the input attributes

![Alt text](overview.png?raw=true "Title")

Getting started
-------------
- Navigate into the `data` folder and run `source get_started.sh`. This will download, extract and preprocess the alligned images from [UTKFaces dataset][1].
- Navigate to `models` folder and run `source get_autoencoders.sh` to download pretrained vanilla autoencoder and adversarially trained autoencoder.
- The `gender_clssifier.py` is the main file to train/evaluate the different classifiers.

Training the gender classifier
-------------
`python gender_claddifier.py --help` would list all the available parameters along with their default value. The defaults should work out of the box for classifier with vanilla AE. `python gender_claddifier.py --remove_race` should 


Evaluating the gender classifiers
-------------
`python gender_claddifier.py --eval` should start the evalutaion with features derived from vanilla autoencoder. `python gender_classifier.py --eval --remove_race` should start the evalutaion with features derived from adversarially trained autoencoder.

[1]:https://susanqq.github.io/UTKFace/
