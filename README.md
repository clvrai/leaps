


# Learning to Synthesize Programs as Interpretable and Generalizable Policies

This project is an implementation of [**Learning to Synthesize Programs as Interpretable and Generalizable Policies**](https://arxiv.org/abs/2108.13643), which is published in [**NeurIPS 2021**](https://neurips.cc/Conferences/2021/). Please visit our [project page](https://clvrai.com/leaps/) for more information.

Neural network policies produced with DRL methods are not human-interpretable and often have difficulty generalizing to novel scenarios. To address these issues, we explore learning structured, programmatic policies. In our framework we learn to synthesize programs solely from reward signals. However, programs are difficult to synthesize purely from environment reward. To this end, we propose a framework Learning Embeddings for lAtent Program Synthesis (LEAPS), which first learns a program embedding space that continuously parameterizes diverse behaviors in an unsupervised manner and then search over the learned program embedding space to yield a program that maximizes the return for a given task.

<p align="center">
    <img src="asset/leaps_model.jpeg"/>
</p>

We evaluate our model on a set of sparse-reward Karel environments---commonly used in the program synthesis domain---specially designed to evaluate the performance differences between our program policies and DRL baselines.

## Environments

### Karel environment
- You can find the codes for the Karel environments in [this directory](./karel_env)

## Getting Started

- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [PyTorch 1.4.0](https://pytorch.org/get-started/previous-versions/#v140)
- Install `virtualenv`, create a virtual environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).

```
pip3 install --upgrade virtualenv
virtualenv prl
source prl/bin/activate
pip3 install -r requirements.txt
```


## Usage

### LEAPS Training
We already include a pre-trained model for LEAPS in this repo, located in `weights/LEAPS/best_valid_params.ptp`. If you want to train an additional model (e.g., with a different seed), follow the steps in Stage 1: Learning program embeddings. Otherwise, to run RL with LEAPS, go directly to Stage 2 instructions.

### Stage 1: Learning program embeddings

- Download dataset from [here](hf.co/datasets/jesbu1/leaps_karel_dataset)

```bash
CUDA_VISIBLE_DEVICES="0" python3 pretrain/trainer.py -c pretrain/cfg.py -d data/karel_dataset/ --verbose --train.batch_size 256 --num_lstm_cell_units 256 --loss.latent_loss_coef 0.1 --rl.loss.latent_rl_loss_coef 0.1 --device cuda:0 --algorithm supervisedRL --optimizer.params.lr 1e-3 --prefix LEAPS
```

- Selected arguments (see the `pretrain/cfg.py` for more details)
    - --configfile/-c: Input file for parameters, constants and initial settings
    - --datadir/-d: dataset directory containing data.hdf5 and id.txt
    - --outdir/-o: Output directory for results
    - --algorithm: supervised, supervisedRL
    - Hyperparameters
        - --train.batch_size: number of programs in one batch during training
        - --num_lstm_cell_units: latent space vector size, rnn hidden layer size
        - --rl.loss.decoder_rl_loss_coef (L<sup>R</sup>): program behavior reconstruction loss
        - --loss.condition_loss_coef (L<sup>L</sup>): latent behavior reconstruction loss
        
- Stage 1 ablations

| Ablation    | Command |
| :--------:  | ---------------------------------------- |
| **LEAPS**   | ```python3 pretrain/trainer.py -c pretrain/cfg.py -d data/karel_dataset/ --verbose --train.batch_size 256 --num_lstm_cell_units 256 --loss.latent_loss_coef 0.1 --rl.loss.latent_rl_loss_coef 0.1 --device cuda:0 --algorithm supervisedRL --optimizer.params.lr 1e-3 --prefix LEAPS``` |
| **LEAPSP**  | ```python3 pretrain/trainer.py -c pretrain/cfg.py -d data/karel_dataset/ --verbose --train.batch_size 256 --device cuda:0 --num_lstm_cell_units 256 --loss.latent_loss_coef 0.1 --loss.condition_loss_coef 0.0 --net.condition.observations initial_state --optimizer.params.lr 1e-3 --prefix LEAPSP``` |
| **LEAPSPR** | ```python3 pretrain/trainer.py -c pretrain/cfg.py -d data/karel_dataset/ --verbose --train.batch_size 256 --num_lstm_cell_units 256 --loss.latent_loss_coef 0.1 --rl.loss.latent_rl_loss_coef 0.1 --device cuda:0 --algorithm supervisedRL --net.condition.freeze_params True --loss.condition_loss_coef 0.0 --optimizer.params.lr 1e-3 --net.condition.observations initial_state --prefix LEAPSPR``` |
| **LEAPSPL** | ```python3 pretrain/trainer.py -c pretrain/cfg.py -d data/karel_dataset/ --verbose --train.batch_size 256 --device cuda:0 --num_lstm_cell_units 256 --loss.latent_loss_coef 0.1 --optimizer.params.lr 1e-3 --prefix LEAPSPL``` |


### Stage 2: CEM search
```bash
python3 pretrain/trainer.py --configfile pretrain/leaps_[leaps_maze/leaps_stairclimber/leaps_topoff/leaps_harvester/leaps_fourcorners/leaps_cleanhouse].py --net.saved_params_path weights/LEAPS/best_valid_params.ptp --save_interval 10 --seed [SEED]
```

- Selected arguments (see the corresponding `pretrain/leaps_*.py` configuration files for more details)
    - Checkpoints: specify the path to a pre-trained checkpoint
        - --net.saved_params_path: load pre-trained parameters (e.g. `weights/LEAPS/best_valid_params.ptp`). If you trained your own stage 1 model, redirect this argument to `best_valid_params.ptp` of that model.
    - Logging:            
        - --save_interval: Save weights at every ith interval (None, int)
    - CEM Hyperparameters:
        - --CEM.population_size: number of programs in one CEM batch
        - --CEM.sigma: CEM sample standard deviation 
        - --CEM.use_exp_sig_decay: boolean variable to indicate exponential decay in sigma
        - --CEM.elitism_rate: percent of the population considered ‘elites’
        - --CEM.reduction: CEM population reduction ['mean', 'weighted_mean']
        - --CEM.init_type: initial distribution to sample from ['normal':N(0,1), 'tiny_normal':N(0,0.1), 'ones':N(1, 0)]

- Results are saved to `pretrain/output_dir`. Use tensorboard to visualize the results. Note: returns are set to a maximum of 1.1, while it is a max of 1.0 in the paper. This reward difference is due to a syntax bonus, simply subtract 0.1 from the printed/tensorboard results to get the corresponding return out of 1.0.
    - By default, if the best program achieves max return (1.1) for 10 CEM iterations in a row, the search process is considered to be converged and it is killed early. The best program and its corresponding return is printed every CEM iteration.
        
## Results

### Stage 1: Learning program embeddings

<p align="center">
    <img src="asset/leaps_acc_P.PNG"/>
</p>

<p align="center">
    <img src="asset/leaps_acc_L.PNG"/>
</p>

### Stage 2: CEM search
- LEAPS performance on different tasks over 5 seeds

    | Task         | Mean Reward |
    | :----------: | ----------- |
    | STAIRCLIMBER |   1.00      |
    | FOURCORNER   |   0.45      |
    | TOPOFF       |   0.81      |
    | MAZE         |   1.00      |
    | CLEANHOUSE   |   0.18      |
    | HARVESTER    |   0.45      |


## Cite the paper

If you find this useful, please cite

```
@inproceedings{
trivedi2021learning,
title={Learning to Synthesize Programs as Interpretable and Generalizable Policies},
author={Dweep Trivedi and Jesse Zhang and Shao-Hua Sun and Joseph J Lim},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=wP9twkexC3V}
}
```

## Authors

[Dweep Trivedi](https://dweeptrivedi.github.io/), [Jesse Zhang](https://jesbu1.github.io/), [Shao-Hua Sun](https://shaohua0116.github.io/)

