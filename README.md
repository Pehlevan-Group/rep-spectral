# GeomNet

Sheng Yang's master thesis: geometrically-regularized neural net

## Running Instruction

Recommended to have ```python >= 3.8``` installed, and make a virtual env

```bash
# create a env
python -m venv ./venv
source ./venv/bin/activate

# install necessary packages
pip3 install numpy pandas scikit-learn matplotlib toml ffmpeg

# install pytorch that is compatible with local build (check out https://pytorch.org)
pip3 install torch torchvision torchaudio
pip3 install tensorboard
```

## Attempt 1: feature map regularization

## Attempt 2: use volume element to prune dataset for subsequent training
