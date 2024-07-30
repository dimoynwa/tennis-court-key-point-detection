import os
import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b1
from torchvision.models.efficientnet import EfficientNet_B1_Weights
import lightning as pl
from datetime import datetime

from visualization import draw_tensor
from .loss import KpLoss
from torchsummary import summary
import cv2

class KeyPointsModel(nn.Module):
    def __init__(self, in_channels=3, num_points=14, dropout=.1,
                 freeze_bacbone=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_points = num_points
        
        self.conv1 = nn.Conv2d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=0)
        
        self.eff_net = efficientnet_b1(EfficientNet_B1_Weights.IMAGENET1K_V2)

        if freeze_bacbone:
            self.freezed_params = []
            for name, param in self.eff_net.named_parameters(prefix='effnet'):
                param.requires_grad_(False)
                self.freezed_params.append(name)

        # (batch_size, 1280, 8, 8)
        self.conv2 = nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1,
                               stride=1, padding=0)
        
        # (batch_size, 512, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=2,
                               stride=2, padding=0)
        
        # (batch_size, 256, 4, 4)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2,
                               stride=2, padding=0)
        self.dropout = nn.Dropout(dropout)
        
        # (batch_size, 256, 2, 2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=num_points*3, kernel_size=2,
                               stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.flatten = nn.Flatten()

        self.l_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.eff_net.features(x)

        x = self.conv2(x)
        x = self.l_relu(x)

        x = self.conv3(x)
        x = self.l_relu(x)

        x = self.conv4(x)
        x = self.l_relu(x)

        x = self.dropout(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        x = self.flatten(x)

        # Reshape
        x = x.view(-1, self.num_points, 3)

        return x
    
class LigthningKeypointsModel(pl.LightningModule):
    def __init__(self, model: KeyPointsModel,
                 img_size=(480, 480),
                 lr=.001,
                 save_val_res_every=5) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.img_size = img_size

        self.loss_fn = KpLoss(.6)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.save_val_res_every = save_val_res_every

        self.base_folder = f'kps-{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
            print(f'Create logs folder {self.base_folder}')

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch=batch, batch_idx=batch_idx)
        self.training_step_outputs.append(loss)
        self.log('train_loss', loss.item(), on_step=True)
        for k, v in loss_dict.items():
            self.log(f'train_{k}', v.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch=batch, batch_idx=batch_idx)
        self.validation_step_outputs.append(loss)
        self.log('val_loss', loss.item(), on_step=True)
        for k, v in loss_dict.items():
            self.log(f'val_{k}', v.item())

        # Evaluate first batch on each epoch divisible by 'save_val_res_every'
        if self.current_epoch > 0 and\
            self.save_val_res_every and\
            self.current_epoch % self.save_val_res_every == 0 and batch_idx == 0:
            self.eval_and_save_results(batch)

        return loss
    
    def common_step(self, batch, batch_idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        inputs, target = batch
        output = self(inputs)
        loss, loss_dict = self.loss_fn(output, target)
        return loss, loss_dict
    
    def on_train_epoch_end(self) -> None:
        all_loss = torch.stack(self.training_step_outputs)
        avg_loss = torch.mean(all_loss)
        self.log('train_loss', avg_loss.item(), on_epoch=True)
        self.training_step_outputs.clear() # free memory

    def on_validation_epoch_end(self) -> None:
        all_loss = torch.stack(self.validation_step_outputs)
        avg_loss = torch.mean(all_loss)
        self.log('val_loss', avg_loss.item(), on_epoch=True)
        self.validation_step_outputs.clear() # free memory

    def print_model_info(self):
        print(f'Train dataloader with {len(self.train_dataloader())} samples.')
        print(f'Validation dataloader with {len(self.val_dataloader())} samples.')
        x, _ = self.train_dataloader()[0]
        summary(self.model, x.shape)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
    
    def eval_and_save_results(self, batch):
        x, _ = batch

        if not os.path.exists(f'{self.base_folder}/Epoch_{self.current_epoch}'):
            os.makedirs(f'{self.base_folder}/Epoch_{self.current_epoch}')
            print(f'Create folder {self.base_folder}/Epoch_{self.current_epoch}')

        predictions: torch.Tensor = self(x)

        for idx, (img, pred) in enumerate(zip(x, predictions)):
            pred[:, 0] *= img.shape[1]
            pred[:, 1] *= img.shape[2]

            frame = draw_tensor(img, pred)
            file_name = f'{self.base_folder}/Epoch_{self.current_epoch}/epoch_{self.current_epoch}_index_{idx}_results.png'
            cv2.imwrite(file_name, frame)
            print(f'Successfully saved image to {file_name}')
