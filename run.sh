#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/countix_generator"

docker build -t ${IMAGE_NAME} "$CURRENT_PATH"/docker && \
docker run -it --rm \
    -v "$CURRENT_PATH"/data:/data \
    -v "$CURRENT_PATH"/src:/src \
    -v "$CURRENT_PATH"/logs:/logs \
    -w /src \
    ${IMAGE_NAME} \
    /bin/bash -c " \
        python main.py \
            --logging \
            --reshape_video \
    "
