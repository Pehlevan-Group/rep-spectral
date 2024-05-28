# rep-spectral: spectral regularization up to feature space

Introduce `rep-spectral`, a spectral regularization up to feature space to encourage learning robust feature representations. Testing with

- Single-hidden-layer on synthetic and MNIST
- ResNet18 on CIFAR10
- BarlowTwins ResNet1 on CIFAR10
- Transfer Learning pretrained on imagenet and finetune on CIFAR10, Stanford Dog, Oxford Flowers, and MIT indoor.

For details, see our preprint [arXiv:2405.17181](https://arxiv.org/abs/2405.17181).

## Running Instruction

Recommended to have ```python >= 3.8``` installed, and make a virtual env

```bash
# create a env
python -m venv ./venv
source ./venv/bin/activate

# install necessary packages
pip3 install numpy pandas scikit-learn matplotlib toml ffmpeg pytest tqdm

# install pytorch that is compatible with local build (check out https://pytorch.org)
# torchvision >= 0.17 (for imagenette api)
pip3 install torch torchvision torchaudio
pip3 install tensorboard
```

To train models, at root, call

```bash
# train a single hidden layer network with
python src/train_reg.py 
```

see detailed arguments inside each script.

To evaluate adversarial performance, run the corresponding scripts started with `eval` at root, for instance

```bash
python src/eval_black_box_robustness.py
```

to evaluate the robustness for single-hidden-layer network.

See a list of bash scripts used in training and evaluation listed under [commands/](commands/). Change memory and runtime configuration accordingly. More detailed running instruction will be provided.

To run unit tests, at root, do the following:

```bash
cd tests/
pytest .
```

## Acknowledgement

If you find this code useful, please consider citing our preprint: 

    @article{yang2024spectral,
      title={Spectral regularization for adversarially-robust representation learning}, 
      author={Yang, Sheng and Zavatone-Veth, Jacob A. and Pehlevan, Cengiz},
      year={2024},
      journal={arXiv preprint},
      eprint={2405.17181},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
