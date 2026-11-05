#!/bin/bash

#export variable for building the image
export HOST_USER_GROUP_ARG=$(id -g $USER)
imagename="clr_calibration:latest"
#build the image
docker build .\
    --tag $imagename \
    --build-arg HOST_USER_GROUP_ARG=$HOST_USER_GROUP_ARG \

