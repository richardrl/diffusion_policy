"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
import datetime

# def ddp_setup(rank: int, world_size: int):
#     """
#     Args:
#         rank: Unique identifier of each process
#        world_size: Total number of processes
#     """
#
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
#     torch.cuda.set_device(rank)
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)
def ddp_setup():
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=300))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)

def main(cfg: OmegaConf):
    try:
        ddp_setup()
    except:
        print("No torch run detected")
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size), nprocs=world_size)