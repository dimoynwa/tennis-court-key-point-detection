import lightning as pl
from typing import Any
from torch.utils.data import DataLoader

class TennisKeypointsDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader) -> None:
        super().__init__()
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

    def setup(self, stage: str) -> None:
        print(f'Setup dataloader for stage: {stage}')
        return super().setup(stage)

    def train_dataloader(self) -> Any:
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl