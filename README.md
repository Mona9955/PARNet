# PARNet
Source code for "Pose-Appearance Relational Modeling for Video Action Recognition"

The pre-trained inception-v2 checkpoint can be downloaded from http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz

environments:

tensorflow: 1.15
python:3.7

training code:

train_concat_add.py

models' code:

inception_v2.py  --> CNN part of SA Module

skel_velocity_model.py  --> TMP Module and PA Module

Model_skel_cnn_vel.py --> PARNet framework
