#!/bin/bash
DEV_CONT_NAME=sphinx
BIONEMO_IMAGE=bionemo-docs

build() {
    docker build -f Dockerfile.docs -t ${BIONEMO_IMAGE} .
}

dev() {
    CMD='bash'
    DOCKER_CMD="docker run --network host "
    DOCKER_CMD="${DOCKER_CMD} -v ${PWD}:/docs --workdir /docs "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/passwd:/etc/passwd:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/group:/etc/group:ro "
    DOCKER_CMD="${DOCKER_CMD} -v /etc/shadow:/etc/shadow:ro "
    DOCKER_CMD="${DOCKER_CMD} -u $(id -u):$(id -g) "
    DOCKER_CMD="${DOCKER_CMD} -v ${HOME}/.ssh:${HOME}/.ssh:ro "
    DOCKER_CMD="${DOCKER_CMD} -v ${PWD}/.vscode-server:${HOME}/.vscode-server:rw "
    set -x
    ${DOCKER_CMD} --rm -it --name ${DEV_CONT_NAME} ${BIONEMO_IMAGE} ${CMD}
    set +x
    exit
}


attach() {
    set -x
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${DEV_CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}

usage () {
                    echo \
"To build, start or attach to BioNemo Documentation Build Container
Example:
    To build BioNeMo Documentation Build Container
        ./launch_docs.sh build
    To start BioNeMo Documentation Build Container for development
        ./launch_docs.sh dev
"

}

case $1 in
    build | dev | attach)
        $@
        ;;
    *)
        usage
        ;;
esac
