import logging
import os

import hydra
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn

from utils.callbacks import IncreaseSequenceLengthCallback
from utils.utils import *
from torch.utils.data import DataLoader
from dataset import NoisyImageDataset
from models.correlation3_unscaled import TrackerNetC

images_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\Noiseimages"      # Provide actual path
events_dir = "D:\DL_Dataset_Fall_2024\Town 01\dvs\aligned_timestamps_data_dl_town01_day\Noiseevents"      # Provide actual path
noise_type = "gaussian"
gaussian_std = 0.02
batch_size = 8
epochs = 10

dataset = NoisyImageDataset(
    images_dir=images_dir,
    events_dir=events_dir,
    noise_type=noise_type,
    gaussian_std=gaussian_std
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Adjust input_channels if needed based on dataset (e.g. if RGB images: 3 for target + 3 for events = 6)
model = TrackerNetC(input_channels=6, noise_training=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

model.train()
for epoch in range(epochs):
    for i, x in enumerate(dataloader):
        # Dummy ground truth: In practice, provide real labels
        ground_truth_y = torch.zeros((x.size(0), 2))

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, ground_truth_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="configs", config_name="train_defaults")
def train(cfg):
    pl.seed_everything(1234)

    # Update configuration dicts with common keys
    propagate_keys(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Instantiate model and dataloaders
    model = hydra.utils.instantiate(
        cfg.model,
        _recursive_=False,
    )
    if cfg.checkpoint_path.lower() != "none":
        # Load weights
        model = model.load_from_checkpoint(checkpoint_path=cfg.checkpoint_path)

        # Override stuff for fine-tuning
        model.hparams.optimizer.lr = cfg.model.optimizer.lr
        model.hparams.optimizer._target_ = cfg.model.optimizer._target_
        model.debug = True
        model.unrolls = cfg.init_unrolls
        model.max_unrolls = cfg.max_unrolls
        model.pose_mode = cfg.model.pose_mode

    data_module = hydra.utils.instantiate(cfg.data)

    # Logging
    if cfg.logging:
        training_logger = pl.loggers.TensorBoardLogger(
            ".", "", "", log_graph=True, default_hp_metric=False
        )
    else:
        training_logger = None

    # Training schedule
    callbacks = [
        IncreaseSequenceLengthCallback(
            unroll_factor=cfg.unroll_factor, schedule=cfg.unroll_schedule
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        devices=[0],
        accelerator="gpu",
        callbacks=callbacks,
        logger=training_logger
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()
