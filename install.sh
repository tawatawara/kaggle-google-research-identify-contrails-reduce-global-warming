# torch/
# # torch 2.0.1
wget "https://download.pytorch.org/whl/cu117/torch-2.0.1%2Bcu117-cp39-cp39-linux_x86_64.whl"
wget "https://download.pytorch.org/whl/cu117/torchaudio-2.0.2%2Bcu117-cp39-cp39-linux_x86_64.whl"
wget "https://download.pytorch.org/whl/cu117/torchvision-0.15.2%2Bcu117-cp39-cp39-linux_x86_64.whl"

poetry env use 3.9.9 
poetry lock
poetry install
poetry update
