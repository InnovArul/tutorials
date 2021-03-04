import pytorch_lightning as pl
from swav_module import SwAV
from pl_bolts.datamodules import STL10DataModule
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)
from pl_bolts.transforms.dataset_normalizations import stl10_normalization

# data
batch_size = 128
dm = STL10DataModule(data_dir='.', batch_size=batch_size, num_workers=8)
dm.train_dataloader = dm.train_dataloader_mixed
dm.val_dataloader = dm.val_dataloader_mixed

dm.train_transforms = SwAVTrainDataTransform(
    normalize=stl10_normalization()
)

dm.val_transforms = SwAVEvalDataTransform(
    normalize=stl10_normalization()
)

# model
model = SwAV(
    lars_wrapper=True,
    online_ft=True,
    gpus=1,
    learning_rate=1e-3,
    num_samples=dm.num_unlabeled_samples,
    gaussian_blur=True,
    queue_length=0,
    dataset='stl10',
    jitter_strength=1.0,
    batch_size=batch_size,
    nmb_prototypes=512,
    nmb_crops=[2,4],
)

# fit
trainer = pl.Trainer(gpus=1, distributed_backend='ddp', num_sanity_val_steps=0,
                    accelerator='ddp')
print("starting trainer")
trainer.fit(model, datamodule=dm)

