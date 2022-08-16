import os 
from torch import optim
from torch import tensor as Tensor
import pytorch_lightning as pl 
import torchvision.utils as vutils
from vae import VAE

class VAEXperiment(pl.LightningModule):
    
    def __init__(self,
    vae_model: VAE,
    params: dict
    ) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(
            *results,
            M_N = self.params['kld_weight'],
            optimizer_idx = optimizer_idx,
            batch_idx = batch_idx
        )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(
            *results,
            M_N = 1.0,
            optimizer_idx = optimizer_idx,
            batch_idx = batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()
    
    def sample_images(self):
        # Get sample reconstruction image

        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
 
        if (self.current_epoch % 1 == 0):
            recons = self.model.generate(test_input, labels = test_label)
            vutils.save_image(
                recons.data,
                os.path.join(self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"
                ),
                #normalize=True,
                nrow=12
            )
        try:
            if (self.current_epoch % 1 == 0):
                samples = self.model.sample(
                    144,
                    self.curr_device,
                    labels=test_label
                )
                vutils.save_image(
                    samples.cpu().data,
                    os.path.join(self.logger.log_dir,
                    "Samples",
                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                    #normalize=True,
                    nrow=12
                )
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )    
        optims.append(optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma= self.params['scheduler_gamma'])
                scheds.append(scheduler)
            return optims, scheds
        except:
            return optims
