import torch
import trimesh
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from models.teethdetr import TeethDETR
from data.st_data import TeethLandDataset
from utils.loss import Criterion


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethDETR(args)
        self.fh_loss = nn.MSELoss()
        self.nch_loss = nn.CrossEntropyLoss()
        self.ch_loss = Criterion(args)

    def forward(self, x):
        return self.net.forward(x)

    def infer(self, x):
        p_xyz, p_labels = self.net.inference(x)
        return p_xyz, p_labels

    def training_step(self, batch, _):
        x, t_idx, mask, g_fheats, g_cheats = batch

        p_fheats, cprobs, p_cheats = self(x)
        fh_loss = self.fh_loss(p_fheats * mask, g_fheats * mask)  # 0.18
        if g_cheats.shape[1] == 0:
            target = torch.zeros([cprobs.shape[0], cprobs.shape[1]], dtype=torch.long).cuda()
            ch_loss = self.nch_loss(cprobs.squeeze(), target.squeeze())  # 0.62
            loss = fh_loss * 10 + ch_loss * 2
        else:
            ch_loss = self.ch_loss(g_cheats, cprobs, p_cheats)
            loss = fh_loss * 10 + ch_loss
        self.log('loss', loss, batch_size=x.size(0))
        self.log('lr', self.optimizers().param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, _):
        x, t_idx, mask, g_fheats, g_cheats = batch

        p_fheats, cprobs, p_cheats = self(x)
        fh_loss = self.fh_loss(p_fheats * mask, g_fheats * mask)
        if g_cheats.shape[1] == 0:
            target = torch.zeros([cprobs.shape[0], cprobs.shape[1]], dtype=torch.long).cuda()
            ch_loss = self.nch_loss(cprobs.squeeze(), target.squeeze())
            loss = fh_loss * 10 + ch_loss * 2
        else:
            ch_loss = self.ch_loss(g_cheats, cprobs, p_cheats)
            loss = fh_loss * 10 + ch_loss
        self.log('val_loss', loss, True, batch_size=x.size(0))

    def test_step(self, batch, _):
        x, t_idx, mask, g_fheats, g_cheats = batch

        p_fheats, cprobs, p_cheats = self(x)
        fh_loss = self.fh_loss(p_fheats * mask, g_fheats * mask)
        if g_cheats.shape[1] == 0:
            target = torch.zeros([cprobs.shape[0], cprobs.shape[1]], dtype=torch.long).cuda()
            ch_loss = self.nch_loss(cprobs.squeeze(), target.squeeze())
            loss = fh_loss * 10 + ch_loss * 2
        else:
            ch_loss = self.ch_loss(g_cheats, cprobs, p_cheats)
            loss = fh_loss * 10 + ch_loss
        self.log('test_loss', loss, True, batch_size=x.size(0))

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        args = self.hparams.args
        return DataLoader(TeethLandDataset(args, args.train_file, True),
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.train_workers,
                          pin_memory=True)

    def val_dataloader(self):
        args = self.hparams.args
        return DataLoader(TeethLandDataset(args, args.val_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.val_workers,
                          pin_memory=True)

    def test_dataloader(self):
        args = self.hparams.args
        return DataLoader(TeethLandDataset(args, args.test_file, False),
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.test_workers,
                          pin_memory=True)


class LitModelInference(LitModel):
    def forward(self, x):
        return torch.argmax(self.net(x), dim=2)
