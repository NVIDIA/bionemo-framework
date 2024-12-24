Environment Setup
===============

While you can install bionemo-moco directly here is an example of specifying the specific torch and cuda versions:

conda create --name moco python=3.10 pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

conda activate moco
python -m pip install 'numpy<2.0'

pip install -r environment/requirements.txt

pre-commit install

pip install -e .
