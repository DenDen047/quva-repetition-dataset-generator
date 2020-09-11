#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/quva_dataset_generator"

data_dir="data"
quva_dir="data/QUVARepetitionDataset"
if [[ ! -d "$quva_dir" ]]; then
    cd ${data_dir} && \
    wget http://isis-data.science.uva.nl/tomrunia/QUVARepetitionDataset.tar.gz && \
    tar -xvf QUVARepetitionDataset.tar.gz
fi

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
    "
