# Score-Matching BED

Code accompanying the paper *Score-Matching for Bayesian Experimental Design*. Trains adaptive design policies for Bayesian experimental design using a learned marginal score network, alongside baselines (PCE/NMC, MLMC, variational marginal / posterior).

## Install

Environment management uses [pixi](https://pixi.prefix.dev/latest/installation/):

```
pixi install -e cuda        # drop -e cuda for CPU only
pixi run -e cuda postinstall
```

Logging uses [wandb](https://wandb.ai/site/) — run `wandb login` once.

## Run

Experiments are [Hydra](https://hydra.cc/) configs in `config/experiment/`. Train a score network and policy on location finding:

```
pixi run -e cuda python run_scripts/run_random_restarts.py experiment=location_finding
```

Other tasks: `cart_pole`, `stoch_pend`, `double_pend`, `gravimetry`. Override any config key on the command line, e.g. `n_restarts=5` or `policy.num_iters=1000`. The variational baselines have their own entry points (`run_var_marg.py`, `run_var_ba.py`) with matching `*_var_marg` / `*_var_post` configs. A quick end-to-end check: `experiment=smoke_test`.
