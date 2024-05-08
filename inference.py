import torch
from torch.utils.data import DataLoader
from oa.dataset import ProcessedTS1x
from oa.trainer.pl_trainer import DDPMModule
from oa.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa.diffusion._normalizer import FEATURE_MAPPING
from pathlib import Path
from tqdm import tqdm
from oa.utils.sampling_tools import (
    write_tmp_xyz,
)
import os
import warnings


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" # prevent OOM
warnings.filterwarnings("ignore", message="An issue occurred while importing 'pyg-lib'.*") # ignore warnings
warnings.filterwarnings("ignore", message="An issue occurred while importing 'torch-sparse'.*") # ignore warnings

# choose device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('test device:', device)

# load checkpoint to pl module
checkpoint_path = './tb_logs/oapainn/37603/checkpoints/ddpm-epoch=3999-val-totloss=408.22.ckpt' # modify this to the path of the checkpoint
filename = 'oapainn' # modify this to the filename of the output
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    map_location=device,
)

# schedule
noise_schedule: str = "polynomial_2"
timesteps: int = 150
precision: float = 1e-5
print('checkpoint:',checkpoint_path, 'file:', filename, 'timestep:', timesteps)

gamma_module = PredefinedNoiseSchedule(
            noise_schedule=noise_schedule,
            timesteps=timesteps,
            precision=precision,
        )
schedule = DiffSchedule(
    gamma_module=gamma_module,
)
ddpm_trainer.ddpm.schedule = schedule
ddpm_trainer.ddpm.T = timesteps
ddpm_trainer = ddpm_trainer.to(device)

# config
batch_size = 16
val_config = dict(
    datadir="oa/data/transition1x/",
    remove_h=False,
    bz=batch_size,
    num_workers=0,
    clip_grad=True,
    gradient_clip_val=None,
    ema=False,
    ema_decay=0.999,
    swapping_react_prod=False,
    append_frag=False,
    use_by_ind=True,
    reflection=False,
    single_frag_only=True,
    only_ts=False,
    lr_schedule_type=None,
    lr_schedule_config=dict(
        gamma=0.8,
        step_size=100,
    ),  # step
)

# dataset
dataset = ProcessedTS1x(Path(val_config['datadir'], 'val.pkl'), **val_config)  # load data
dataloader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=dataset.collate_fn)

# eval

ddpm_trainer.ddpm.eval()
interations = 64 # modify this to the number of genrations
for iteration in range(interations):
    ex_ind = 0
    os.makedirs(f"./generation/{filename}/iter_{iteration}", exist_ok=True)
    for idx, batch in enumerate(tqdm(dataloader)):
        if idx == 200:
            break
        # inpaint
        representations, conditions = batch
        conditions = conditions.to(device)
        xh_fixed = [
            torch.cat(
                [repre[feature_type] for feature_type in FEATURE_MAPPING],
                dim=1,
            ).to(device)
            for repre in representations
        ]
        n_samples = representations[0]["size"].size(0)
        fragments_nodes = [repre["size"].to(device) for repre in representations]
        with torch.no_grad():
            out_samples, _ = ddpm_trainer.ddpm.inpaint(
                n_samples=n_samples,
                fragments_nodes=fragments_nodes,
                conditions=conditions,
                return_frames=1,
                resamplings=5,
                jump_length=5,
                timesteps=None,
                xh_fixed=xh_fixed,
                frag_fixed=[0,],
            )

        write_tmp_xyz(
        fragments_nodes, 
        out_samples[0], 
        idx=[0, 1, 2], 
        localpath=f"./generation/{filename}/iteration{iteration}",
        ex_ind=ex_ind,
        )

        trajs = {}
        if idx % 50 == 0:
            print(f"iter {iteration} batch {idx} is done.")
            traj_path = './generation/' + filename + '/iteration' + str(iteration)
            trajs[f'batch{idx}'] = [out_samples, fragments_nodes]
        
        # save torch file
        torch.save(trajs, traj_path + '/trajs.pth')
            
        ex_ind += len(fragments_nodes[0])
        



    
    
