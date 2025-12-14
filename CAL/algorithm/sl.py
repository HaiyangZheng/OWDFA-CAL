import torch.nn.functional as F
import lightning.pytorch as pl

import wandb
from loguru import logger
from timm.models import resume_checkpoint

from CAL.utils import gather_tensor, val_stat
import CAL.encoder as encoder
import CAL.optimizers as optimizers
import CAL.schedulers as schedulers


class SLModel(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()

        self.args = args
        # construct the encoder
        self.encoder = encoder.__dict__[args.model.name](**vars(args.model.params))
        if args.model.resume is not None:
            resume_checkpoint(self.encoder, args.model.resume)
            if args.local_rank == 0:
                logger.info(f'resume model from {args.model.resume}')

    def configure_optimizers(self):
        optimizer = optimizers.__dict__[self.args.optimizer.name](
            self.encoder.parameters(), **vars(self.args.optimizer.params))

        if self.args.scheduler.name == "None":
            return optimizer
        else:
            scheduler = schedulers.__dict__[self.args.scheduler.name](
                optimizer, **vars(self.args.scheduler.params))
            return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pass


    def on_validation_epoch_start(self):
        self.val_step_outputs = {
            'preds': [],
            'label': [],
            'conf': [],
        }

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        targets = batch['target']

        output = self.encoder(images)
        prob = F.softmax(output, dim=1)
        conf, pred = prob.max(1)

        self.val_step_outputs['preds'].extend(pred)
        self.val_step_outputs['label'].extend(targets)
        self.val_step_outputs['conf'].extend(conf)
        return pred

    def on_validation_epoch_end(self):
        logger.info(f'Epoch-{self.current_epoch} validation finished')

        y_pred = gather_tensor(self.val_step_outputs['preds'], dist_=False, to_numpy=True).astype(int)
        y_label = gather_tensor(self.val_step_outputs['label'], dist_=False, to_numpy=True).astype(int)
        y_conf = gather_tensor(self.val_step_outputs['conf'], dist_=False, to_numpy=True)

        results = val_stat(y_pred, y_label, y_conf, self.num_known)
        self.log_dict(results, on_epoch=True)
        logger.info(f"val_stat: {results}")

        try:
            if self.args.use_wandb:
                wandb.log(results, step=self.current_epoch)
        except:
            pass

    def predict_step(self, batch, batch_idx):
        images = batch['image']
        feature, _ = self.encoder.forward_features(images)
        feature = feature.detach().cpu().numpy()

        return {
            'feature': feature,
            'target': batch['target'],
            'img_path': batch['img_path'],
        }
