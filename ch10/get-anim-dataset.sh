#!/usr/bin/env bash

mkdir -p AnimTransDistr
cd AnimTransDistr
wget http://vision.stanford.edu/Datasets/AnimTransDistr.rar
unrar x AnimTransDistr.rar
# The following file is a weird file:
rm Anims/104034.jpg
