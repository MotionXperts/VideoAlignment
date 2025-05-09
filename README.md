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

### Prepare
#### FS dataset
There are four types of figure skating actions included in the FS dataset. Each action corresponds to a specific configuration file used for training or evaluation purposes.

Action Types and Corresponding Configuration File Paths:
**Axel_single_jump**
- This directory is used to train the action types 'Axel' and 'Axel_com'.
- The skater in the video performs a single jump.
- config path : `./result/Axel_single_jump/config.yaml`

**Axel_double_jump**
- This directory is used to train the action types 'Axel' and 'Axel_com'.
- The skater in the video performs a double jump.
- config path : `./result/Axel_double_jump/config.yaml`

**Lutz_double_jump**
- This directory is used to train the action type 'Lutz'.
- The skater in the video performs a double jump.
- config path : `./result/Lutz_double_jump/config.yaml`

**Loop_double_jump**
- This directory is used to train the action type 'Loop'
- The skater in the video performs a double jump.
- config path : `./result/Loop_double_jump/config.yaml`