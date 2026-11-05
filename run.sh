#!/bin/bash

#parameters
containerName="clr_calibration"
imageName="clr_calibration"
imageTag=latest

src_folder=$(pwd)/calibration
# Host folder holding the datasets and checkpoints, mounted at /home/appuser/data in the container.
# Override with: CLR_DATA=/path/to/data ./run.sh
data_folder=${CLR_DATA:-$(pwd)/data}

xhost local:root
XAUTH=/tmp/.docker.xauth


#Set permissions for newly created files. All new files will have the following permissions.
#Important is that the group permissions of the files created are set to read and write.
#We add src as a volume, so we will be able to edit and delete the files created in the container.
#Output of this command is silenced
setfacl -PRdm u::rwx,g::rwx,o::r ./ &> /dev/null


#check if container exists
if [[ $( docker ps -a -f name=' '$containerName | grep -w ' '$containerName | wc -l ) -eq 1 ]];
then
    echo "Container already exists. Do you want to restart it or remove it?"
    select yn in "Restart" "Remove"; do
        case $yn in
            Restart )
                echo "Restarting it...";
                docker restart $containerName;
                break;;
            Remove )
                echo "Stopping it and deleting it... ";
                docker stop $containerName;
                docker rm $containerName;
                break;;
        esac
    done

else
    docker run \
        --env DISPLAY=${DISPLAY} \
        --env="QT_X11_NO_MITSHM=1" \
        --env="XAUTHORITY=$XAUTH" \
        --env NVIDIA_VISIBLE_DEVICES=all \
        --env NVIDIA_DRIVER_CAPABILITIES=all \
        --volume $src_folder:/home/appuser/src/calibration/ \
        --volume $data_folder:/home/appuser/data/ \
        --volume /tmp/.X11-unix:/tmp/.X11-unix \
        --volume="$XAUTH:$XAUTH" \
        --network host \
        --tty \
        --detach \
        --gpus all \
        --runtime=nvidia \
        --name $containerName \
        --ipc=host \
        --ulimit nofile=65536:65536 \
        $imageName:$imageTag 
fi

exit 0
