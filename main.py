import logging

import torch
from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg, load_cfg,
                                             set_out_dir, set_run_dir)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device

# Change if precision need improvements
torch.set_float32_matmul_precision('medium')
# Levels = 'info' | 'prod'
log_level = 'info'
# Load cmd line args
args = parse_args()
# Load config file
load_cfg(cfg, args)
set_out_dir(cfg.out_dir, args.cfg_file)
# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)
dump_cfg(cfg)
# Repeat for different random seeds
set_run_dir(cfg.out_dir)
set_printing()
# Set configurations for each run
seed_everything(cfg.seed)
auto_select_device()
# Set machine learning pipeline
datamodule = GraphGymDataModule()
model = create_model()
# Print model info
if log_level == 'info':
  logging.info(model)
  logging.info(cfg)
cfg.params = params_count(model)
if log_level == 'info':
  logging.info('Num parameters: %s', cfg.params)
train(model, datamodule, logger=True)