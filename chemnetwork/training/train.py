###############################################################
#
#   Script for training the flow matching network.
#
###############################################################

import torch
import hydra
import logging
import wandb

from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from chemnetwork.models.flow_matching import FlowModel, FlowMatcher, MSELoss
from chemnetwork.data.point_clouds import PointCloudGenerator
from chemnetwork.utils import set_seed


@hydra.main(version_base=None, config_path="../../preferences", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main method that starts the training of a flow matching model.

    Args:
        cfg (DictConfig): Hydra config dictionary.
    """

    log = logging.getLogger(__name__)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_path = run_dir / Path(cfg.train_parameters.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.train_parameters.random_seed)
    log.info(f"Using device {device}.")

    # Initialize wandb run
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.train_parameters.comment,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Initialize data generator
    data_generator = PointCloudGenerator(
        num_samples=cfg.data.train_batch_size,
        target_name=cfg.data.target_distribution,
        target_noise=cfg.data.target_data_noise
    )
    log.info(f"Initialized data generator with target distribution '{cfg.data.target_distribution}' and batch size {cfg.data.train_batch_size}.")

    # Initialize model
    model = FlowModel(
        in_features=cfg.model.params.in_features,
        hidden_features=cfg.model.params.hidden_features,
        out_features=cfg.model.params.out_features
    )
    log.info(f"Initialized model {cfg.model.name}.")

    # Initialize loss function
    criterion = MSELoss()

    # Initialize optimizer
    optim = torch.optim.AdamW(params=model.parameters(), lr=cfg.train_parameters.learning_rate)

    # Initialize flow matching schema
    flow_matcher = FlowMatcher()

    # Traingin loop
    for step in tqdm(range(0, cfg.train_parameters.train_loop_iterations), leave=False):
        
        # Reset optimizer gradients
        optim.zero_grad()   # NOTE: Why this again in every loop iteration?
        
        # Draw data from p_0 and p_1
        x0 = data_generator.draw_source()
        x1 = data_generator.draw_target()

        # Compute interpolant and conditional flow
        t, xt, ut = flow_matcher.sample_interpolant_and_target(x_0=x0, x_1=x1)

        # Network prediction
        vt_pred = model(x_in=xt, t=t)

        # Compute loss
        loss_val = criterion(u_t=ut, v_hat=vt_pred)

        # Log the loss
        wandb.log({"train/loss": loss_val.item()}, step=step)
        if cfg.train_parameters.verbose and (step != 0) and (step % 500 == 0):
            log.info(f"Loss: {loss_val:.3f}")
        
        # Save checkpoint
        if (step !=0) and (step % cfg.train_parameters.save_every == 0):
            checkpoint = {
                'model': model.state_dict(),
                'optim': optim.state_dict(),
                'step': step,
                'config': OmegaConf.to_container(cfg, resolve=True),
            }
            checkpoint_name = checkpoint_path / Path(f"{cfg.model.name}_{cfg.data.target_distribution}_weights_step_{step}.pt")
            torch.save(checkpoint, checkpoint_name)
            log.info(f"Saved model as {checkpoint_name}")
        
        # Update
        loss_val.backward()
        optim.step()

    # Final save
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'step': step,
        'config': OmegaConf.to_container(cfg, resolve=True),
    }
    checkpoint_name = checkpoint_path / Path(f"{cfg.model.name}_{cfg.data.target_distribution}_weights_step_{step}.pt")
    torch.save(checkpoint, checkpoint_name)
    log.info(f"Saved model as {checkpoint_name}")
    
    # Create wandb artifacts    # TODO: Save this every_nth step as well?
    artifact = wandb.Artifact(name=f"{cfg.model.name}_{cfg.data.target_distribution}_weights", type="model")
    artifact.add_file(str(checkpoint_name))
    run.log_artifact(artifact)

    # Close the run
    run.finish()
        

if __name__ == "__main__":

    # Start the process
    main()