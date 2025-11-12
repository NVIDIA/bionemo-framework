To build the software compatibility testing image, perform:
```bash
CUDA='12.4.1'
PYTHON='3.10.12'
TORCH='2.4.0'
CUDNN_VERSION=''  # make this '8' iff using < CUDA 12.4
time docker buildx build \
    -f ci/docker/Dockerfile.conda \
    --build-arg CUDA_VERSION=${CUDA} \
    --build-arg PYTHON_VERSION=${PYTHON} \
    --build-arg PYTORCH_VERSION="${TORCH}" \
    --build-arg CUDNN_VERSION="${CUDNN_VERSION}" \
    -t "bionemo--cuda_$(echo $CUDA | sed s/'\.'/'_'/g)--python_$(echo $PYTHON | sed s/'\.'/'_'/g)--pytorch_$(echo $TORCH | sed s/'\.'/'_'/g):$(git rev-parse HEAD)" .
```
