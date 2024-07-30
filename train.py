import argparse
from torch.utils.data import DataLoader
from data import validate_dataset
from data.data_module import TennisKeypointsDataModule
from data.dataset import KeypointsDataset
from data.transforms import TennisKeypointsTransforms
from model import KeyPointsModel
from model.model import LigthningKeypointsModel
from util.constants import DATA_TRAIN_FILE_NAME, DATA_VAL_FILE_NAME, IMAGES_FOLDER
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True, help='Data folder')
    parser.add_argument("--imgsz", required=False, type=int, default=480)
    parser.add_argument("--rotation", required=False, type=int, default=35)
    parser.add_argument("--validate_ds", required=False, type=bool, default=True)
    parser.add_argument("--batch_size", required=False, type=int, default=8)
    parser.add_argument("--freeze_backbone", required=False, type=bool, default=True)
    parser.add_argument("--val_every_n_epochs", required=False, type=int, default=5)
    parser.add_argument("--lr", required=False, type=float, default=.001)
    parser.add_argument("--log_every_n_steps", required=False, type=int, default=5)
    parser.add_argument("--epochs", required=False, type=int, default=False)

    args = parser.parse_args()
    print(f'Args: {args}')

    image_folder = f'{args.data_dir}/{IMAGES_FOLDER}'

    train_data_file = f'{args.data_dir}/{DATA_TRAIN_FILE_NAME}'
    train_transformations = TennisKeypointsTransforms(img_size=args.imgsz, rotate=args.rotation,
                                                      normalize=True, keypoints_format='xy')

    train_ds = KeypointsDataset(train_data_file, image_folder,
                                transfrormations=train_transformations)
    
    print(f'Train dataset created with {len(train_ds)} samples, image size: {args.imgsz},\
           rotation: {args.rotation}')

    val_data_file = f'{args.data_dir}/{DATA_VAL_FILE_NAME}'
    val_transformations = TennisKeypointsTransforms(img_size=args.imgsz, rotate=0,
                                                    normalize=True, keypoints_format='xy')
    val_ds = KeypointsDataset(val_data_file, image_folder, 
                              transfrormations=val_transformations)
    
    print(f'Validation dataset created with {len(train_ds)} samples, image size: {args.imgsz}')

    if args.validate_ds:
        validate_dataset(train_ds, 'Train')
        validate_dataset(val_ds, 'Validation')

    # Creating dataloaders
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    kpm = KeyPointsModel(freeze_bacbone=args.freeze_backbone)
    pl_model = LigthningKeypointsModel(kpm, save_val_res_every=args.val_every_n_epochs,
                                       lr=args.lr)

    # Data module
    data_module = TennisKeypointsDataModule(train_dataloader=train_dl,
                                            val_dataloader=val_dl)
    
    # Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'{pl_model.base_folder}',
        filename='kp-tennis-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=5
    )

    csv_logger = CSVLogger(save_dir=f'{pl_model.base_folder}')

    trainer = Trainer(
        logger=csv_logger,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback])

    trainer.fit(pl_model, datamodule=data_module)

