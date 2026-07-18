###############################################################
#
#   Script for training the flow matching network.
#
###############################################################

import torch
import hydra
import logging

from tqdm import tqdm
from omegaconf import DictConfig

from chemnetwork.models.flow_matching import FlowModel, FlowMatcher, MSELoss
from chemnetwork.data.point_clouds import PointCloudGenerator


@hydra.main(version_base=None, config_path="../../preferences", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main method that starts the training of a flow matching model.

    Args:
        cfg (DictConfig): Hydra config dictionary.
    """

    log = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device {device}.")

    # Initialize data generator
    data_generator = PointCloudGenerator(num_samples=cfg.train_parameters.train_batch_size)
    log.info(f"Initialized data generator with batch size {cfg.train_parameters.train_batch_size}.")

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
        
        # Reser optimizer gradients
        optim.zero_grad()   # NOTE: Why this again in every loop iteration?
        
        # Draw data from p_0 and p_1
        x0 = data_generator.draw_x0()
        x1 = data_generator.draw_x1()

        # Compute interpolant and conditional flow
        t, xt, ut = flow_matcher.sample_interpolant_and_target(x_0=x0, x_1=x1)

        # Network prediction
        vt_pred = model(x_in=xt, t=t)

        # Compute loss
        loss_val = criterion(u_t=ut, v_hat=vt_pred)
        
        if (step != 0) and (step % 500 == 0):
            log.info(f"Loss: {loss_val:.3f}")
        
        # Update
        loss_val.backward()
        optim.step()
        

if __name__ == "__main__":

    # Start the process
    main()