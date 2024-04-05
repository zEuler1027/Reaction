from typing import Dict, List, Optional, Tuple

from pathlib import Path
import torch
import copy
from torch import nn
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from pytorch_lightning import LightningModule


from oa.dataset import (
    ProcessedTS1x,
)
from oa.dynamics import EGNNDynamics
from oa.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from oa.diffusion._normalizer import FEATURE_MAPPING
from oa.diffusion.en_diffusion import EnVariationalDiffusion
from oa.trainer._metrics import average_over_batch_metrics, pretty_print
import oa.utils.training_tools as utils
from oa.analyze.rmsd import batch_rmsd

PROCESS_FUNC = {
    "TS1x": ProcessedTS1x,
    "OC20": None,
    "CH4": None,
}
FILE_TYPE = {
    "TS1x": ".pkl",
    "OC20": ".pkl",
    "CH4": ".pkl",
}
LR_SCHEDULER = {
    "cos": CosineAnnealingWarmRestarts,
    "step": StepLR,
}


class DDPMModule(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        optimizer_config: Dict,
        training_config: Dict,
        node_nfs: List[int] = [9] * 3,
        edge_nf: int = 4,
        condition_nf: int = 3,
        fragment_names: List[str] = ["inorg_node", "org_edge", "org_node"],
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        noise_schedule: str = "polynomial_2",
        timesteps: int = 1000,
        precision: float = 1e-5,
        loss_type: str = "l2",
        pos_only: bool = False,
        process_type: Optional[str] = None,
        model: nn.Module = None,
        enforce_same_encoding: Optional[List] = None,
        scales: List[float] = [1.0, 1.0, 1.0],
        eval_epochs: int = 20,
        source: Optional[Dict] = None,
        fixed_idx: Optional[List] = None,
    ) -> None:
        super().__init__()
        egnn_dynamics = EGNNDynamics(
            model_config=model_config,
            node_nfs=node_nfs,
            edge_nf=edge_nf,
            condition_nf=condition_nf,
            fragment_names=fragment_names,
            pos_dim=pos_dim,
            update_pocket_coords=update_pocket_coords,
            condition_time=condition_time,
            edge_cutoff=edge_cutoff,
            model=model,
            enforce_same_encoding=enforce_same_encoding,
            source=source,
        )


        gamma_module = PredefinedNoiseSchedule(
            noise_schedule=noise_schedule,
            timesteps=timesteps,
            precision=precision,
        )
        schedule = DiffSchedule(gamma_module=gamma_module)

        self.ddpm = EnVariationalDiffusion(
            dynamics=egnn_dynamics,
            schdule=schedule,
            size_histogram=None,
            loss_type=loss_type,
            pos_only=pos_only,
            fixed_idx=fixed_idx,
        )
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.training_config = training_config
        self.loss_type = loss_type
        self.n_fragments = len(fragment_names)
        self.remove_h = training_config["remove_h"]
        self.pos_only = pos_only
        self.process_type = process_type or "QM9"
        self.scales = scales

        sampling_gamma_module = PredefinedNoiseSchedule(
            noise_schedule="polynomial_2",
            timesteps=150,
            precision=precision,
        )
        self.sampling_schedule = DiffSchedule(gamma_module=sampling_gamma_module)
        self.eval_epochs = eval_epochs

        self.clip_grad = training_config["clip_grad"]
        if self.clip_grad:
            self.gradnorm_queue = utils.Queue()
            self.gradnorm_queue.add(3000)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.ddpm.parameters(), **self.optimizer_config) # default
        if not self.training_config["lr_schedule_type"] is None:
            scheduler_func = LR_SCHEDULER[self.training_config["lr_schedule_type"]]
            scheduler = scheduler_func(
                optimizer=optimizer, **self.training_config["lr_schedule_config"]
            )
            return [optimizer], [scheduler]
        else:
            return optimizer

    def setup(self, stage: Optional[str] = None):
        func = PROCESS_FUNC[self.process_type] # "TS1x": ProcessedTS1x
        ft = FILE_TYPE[self.process_type] # "TS1x": ".pkl"
        if stage == "fit":
            self.train_dataset = func(
                Path(self.training_config["datadir"], f"train{ft}"),
                **self.training_config,
            )
            self.training_config["reflection"] = False  # Turn off reflection in val.
            self.val_dataset = func(
                Path(self.training_config["datadir"], f"val{ft}"),
                **self.training_config,
            )
        elif stage == "test":
            self.test_dataset = func(
                Path(self.training_config["datadir"], f"test{ft}"),
                **self.training_config,
            )
        else:
            raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.training_config["bz"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.training_config["bz"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            collate_fn=self.test_dataset.collate_fn,
        )

    def compute_loss(self, batch):
        representations, conditions = batch
        loss_terms = self.ddpm.forward(
            representations,
            conditions,
        )
        info = {}
        if not self.pos_only:
            denoms = [
                (self.ddpm.pos_dim + self.ddpm.node_nfs[ii])
                * representations[ii]["size"]
                for ii in range(self.n_fragments)
            ]
        else:
            denoms = [
                self.ddpm.pos_dim * representations[ii]["size"]
                for ii in range(self.n_fragments)
            ]
        error_t_normalized = [
            loss_terms["error_t"][ii] / denoms[ii] * self.scales[ii]
            for ii in range(self.n_fragments)
        ]
        if self.loss_type == "l2" and self.training:
            # normalize loss_t
            loss_t = torch.stack(error_t_normalized, dim=0).sum(dim=0)

            # normalize loss_0
            loss_0_x = [
                loss_terms["loss_0_x"][ii]
                * self.scales[ii]
                / (self.ddpm.pos_dim * representations[ii]["size"])
                for ii in range(self.n_fragments)
            ]
            loss_0_x = torch.stack(loss_0_x, dim=0).sum(dim=0)
            loss_0_cat = torch.stack(loss_terms["loss_0_cat"], dim=0).sum(dim=0)
            loss_0_charge = torch.stack(loss_terms["loss_0_charge"], dim=0).sum(dim=0)
            loss_0 = loss_0_x + loss_0_cat + loss_0_charge

        # VLB objective or evaluation step
        else:
            # Note: SNR_weight should be negative
            error_t = [
                -self.ddpm.T * 0.5 * loss_terms["SNR_weight"] * _error_t
                for _error_t in loss_terms["error_t"]
            ]
            loss_t = torch.stack(error_t, dim=0).sum(dim=0)

            loss_0_x = torch.stack(loss_terms["loss_0_x"], dim=0).sum(dim=0)
            loss_0_cat = torch.stack(loss_terms["loss_0_cat"], dim=0).sum(dim=0)
            loss_0_charge = torch.stack(loss_terms["loss_0_charge"], dim=0).sum(dim=0)
            loss_0 = (
                loss_0_x + loss_0_cat + loss_0_charge
            )

        nll = loss_t + loss_0 
        # nll = loss_t

        for ii in range(self.n_fragments):
            info[f"error_t_{ii}"] = error_t_normalized[ii].mean().item() / (
                self.scales[ii] + 1e-4
            )

        return nll, info

    def eval_inplaint_batch(
        self,
        batch: List,
        resamplings: int = 5,
        jump_length: int = 5,
        frag_fixed: List = [0, 2],
    ):
        sampling_ddpm = copy.deepcopy(self.ddpm)
        sampling_ddpm.schedule = self.sampling_schedule
        sampling_ddpm.T = self.sampling_schedule.gamma_module.timesteps
        sampling_ddpm.eval()

        representations, conditions = batch
        xh_fixed = [
            torch.cat(
                [repre[feature_type] for feature_type in FEATURE_MAPPING],
                dim=1,
            )
            for repre in representations
        ]
        n_samples = representations[0]["size"].size(0)
        fragments_nodes = [repre["size"] for repre in representations]
        with torch.no_grad():
            out_samples, _ = sampling_ddpm.inpaint(
                n_samples=n_samples,
                fragments_nodes=fragments_nodes,
                conditions=conditions,
                return_frames=1,
                resamplings=resamplings,
                jump_length=jump_length,
                timesteps=None,
                xh_fixed=xh_fixed,
                frag_fixed=frag_fixed,
            )
        rmsds = batch_rmsd(
            fragments_nodes,
            out_samples[0],
            xh_fixed,
            idx=1,
            threshold=0.5,
        )
        return np.mean(rmsds), np.median(rmsds)

    def training_step(self, batch, batch_idx):
        nll, info = self.compute_loss(batch)
        loss = nll.mean(0)

        for k, v in info.items():
            self.log(f"train-{k}", v, rank_zero_only=True)

        if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
            if self.trainer.is_global_zero:
                print(
                    "evaluation on samping for training batch...",
                    batch[1].shape,
                    batch_idx,
                )
            rmsd_mean, rmsd_median = self.eval_inplaint_batch(batch)
            info["rmsd"], info["rmsd-median"] = rmsd_mean, rmsd_median

        info["loss"] = loss
        return info

    def _shared_eval(self, batch, batch_idx, prefix, *args):
        nll, info = self.compute_loss(batch)
        loss = nll.mean(0)
        info["totloss"] = loss.item()

        if (self.current_epoch + 1) % self.eval_epochs == 0 and batch_idx == 0:
            if self.trainer.is_global_zero:
                print(
                    "evaluation on samping for validation batch...",
                    batch[1].shape,
                    batch_idx,
                )
            info["rmsd"], info["rmsd-median"] = self.eval_inplaint_batch(batch)


        info_prefix = {}
        for k, v in info.items():
            info_prefix[f"{prefix}-{k}"] = v
        return info_prefix

    def validation_step(self, batch, batch_idx, *args):
        torch.cuda.empty_cache() 
        return self._shared_eval(batch, batch_idx, "val", *args)

    def test_step(self, batch, batch_idx, *args):
        return self._shared_eval(batch, batch_idx, "test", *args)

    def validation_epoch_end(self, val_step_outputs):
        val_epoch_metrics = average_over_batch_metrics(val_step_outputs)
        if self.trainer.is_global_zero:
            pretty_print(self.current_epoch, val_epoch_metrics, prefix="val")
        val_epoch_metrics.update({"epoch": self.current_epoch})
        for k, v in val_epoch_metrics.items():
            self.log(k, v, sync_dist=True)

    def training_epoch_end(self, outputs) -> None:
        epoch_metrics = average_over_batch_metrics(
            outputs, allowed=["rmsd", "rmsd-median"]
        )
        if 'rmsd' in epoch_metrics:
            self.log("train-rmsd", epoch_metrics["rmsd"], sync_dist=True)
            self.log("train-rmsd-median", epoch_metrics["rmsd-median"], sync_dist=True)
        print('max_memory_allocated:{}'.format(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024))
        torch.cuda.empty_cache() 

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 1.5 * stdev of the recent history.
        # max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std()
        max_grad_norm = 1.5 * self.gradnorm_queue.mean() + 3 * self.gradnorm_queue.std() # modified
        # Get current grad_norm
        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = utils.get_grad_norm(params)

        # Lightning will handle the gradient clipping
        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

        if float(grad_norm) > max_grad_norm:
            self.gradnorm_queue.add(float(max_grad_norm))
        else:
            self.gradnorm_queue.add(float(grad_norm))

        if float(grad_norm) > max_grad_norm:
            print(
                f"Clipped gradient with value {grad_norm:.1f} "
                f"while allowed {max_grad_norm:.1f}"
            )
