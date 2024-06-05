# rep-spectral: spectral regularization up to feature space

Introduce `rep-spectral`, a spectral regularization up to feature space to encourage learning robust feature representations. Testing with

- Single-hidden-layer on synthetic and MNIST
- ResNet18 on CIFAR10
- BarlowTwins ResNet1 on CIFAR10
- Transfer Learning pretrained on imagenet and finetune on CIFAR10, Stanford Dog, Oxford Flowers, and MIT indoor.

For details, see our preprint [arXiv:2405.17181](https://arxiv.org/abs/2405.17181).

## Environment Setup

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

## Running Instruction

For each model, we first train by calling scripts in [src/](src/) that starts with `train_*` and evaluate adversarial robustness by running scripts that start with `eval_black_box_robustness_*`.

### Path Setup

We first specify the relative paths with respect to the root of the project in [config.toml](comfig.toml):

- `data_dir`: path to read data;
- `model_dir`: path to store model parameters;
- `result_dir`: path to dump Kfold train test results.

along with the associated tags. For experiment tag, see the following snippet also in [config.toml](config.toml)

```toml
["exp"]
data_dir='data/'
model_dir='model/'
result_dir='result/'
```

### Model Training & Evaluation

For example, to train ResNet18 on cifar10 with rep-spectral regularization, $\lambda = 0.01$, burnin period 160, eigen-direction update amortized across 24 parameter updates, and other parameters, we call the following at root

```bash
python src/train_reg_resnet.py  --model 18 --lr 0.01 --epochs 200 --wd 1e-4 --batch-size 1024 --tag exp --log-model 20 --log-epoch 5 --lam 0.01 --reg eig-ub --burnin 160 --reg-freq-update 24
```

where `log-model` is the frequency measured in epochs to log model parameters, and `log-epoch` is the frequency to report statistics on test set. here we use `eig-ub` to flag using rep-spectral. See detailed arguments inside each script.

To evaluate adversarial performance, run the corresponding scripts started with `eval` at root, for instance, to evaluate the above trained model,

```bash
python src/eval_black_box_robustness.py --model 18 --lr 0.01 --epochs 200 --wd 1e-4 --batch-size 1024 --tag exp --log-model 20 --log-epoch 5 --lam 0.01 --reg eig-ub --burnin 160 --reg-freq-update 24 --vmin -3 --vmax 3 --eval-epoch $epochs --eval-sample-size 1000
```

with the same list of arguments along with other evaluation arguments, such as the range of the perturbed image (within [-3, 3] since cifar10 are normalized by default), and the number of test samples to evaluate (1000 in the above case).

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
