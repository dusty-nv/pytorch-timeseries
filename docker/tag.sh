#!/usr/bin/env bash

# find OS version
source docker/l4t_version.sh

if [ $L4T_VERSION = "32.5.1" ]; then
	CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
elif [ $L4T_VERSION = "32.5.0" ]; then
	CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r32.5.0-py3"
elif [ $L4T_VERSION = "32.4.4" ]; then
	CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.4-py3"
elif [ $L4T_VERSION = "32.4.3" ]; then
	CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.3-py3"
elif [ $L4T_VERSION = "32.4.2" ]; then
	CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.2-py3"
fi
#else
#	echo "cannot find l4t-ml docker container for L4T R$L4T_VERSION"
#	echo "please upgrade your version of JetPack"
#	exit 1
#fi

