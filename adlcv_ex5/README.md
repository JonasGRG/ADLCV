# nerf-project

## Installation
* `conda create -n nerf python=3.10.13`
* `conda activate nerf`
* `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
* `conda install numpy matplotlib tqdm pyyaml opencv tensorboard imageio gdown`
* `python download_data.py`

## Notes
* If the rendering process takes too long, reduce the `TARGET_SIZE` (`playground.py` line 75)
* You will train with images $50\times 50$ and a smaller network. Therefore, the results of the pre-trained model will be better.
* The default scene of `train.py` is "chair". To train on a different scene: `python train.py --scene-name <name>`. Each config file corresponds to a different scene name.
* You can change the hyperparameters of your model from the config file.
* IMPORTANT: do not change neither the layers names of the `NeRF` mdoel nor the scene name at the `playground.py`
## Logging
* The code uses tensorboard to log the train loss. Use the command `tensorboard --logdir=runs` to observe the training loss.
