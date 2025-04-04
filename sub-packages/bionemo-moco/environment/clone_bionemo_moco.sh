git clone --filter=blob:none --sparse https://github.com/NVIDIA/bionemo-framework.git
cd bionemo-framework
git sparse-checkout set sub-packages/bionemo-moco
mv sub-packages/bionemo-moco ..
cd ..
rm -rf bionemo-framework
