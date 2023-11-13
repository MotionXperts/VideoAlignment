## This repository is created by U An
In this repo, I aim to fuse the method of PR-VIPE and CARL into one model
To be specific, the original pipelin of CARL is:
ResNet50 -> Transformer -> Projection
which, in ResNet, captures the RGB features instead of skeleton data

Google's tcc also uses ResNet to capture RGB feature

The differences between these 2 are the followings:

1. Tcc uses 2 video pairs to train the model / CARL uses 1 video and augment it into 2 subvideos
2. Tcc utilize cycle back consitency and contrasitive loss as loss functions / CARL compare the KL-divergence between similarity of the 2 augmented videos' timestamp and Guassion distribution (Both take adjacent frames into account.)

Since there's little point in augmenting skeleton if we uses PR-VIPE. 
I am going to implement the following structure:

Raw Video -> Alphapose -> Skeleton information -> PR-VIPE -> Transformer Encoder -> (embeddings) 

The weights in Alphapose and PR-VIPE does not get updated, we only update weights in transformer encoder
