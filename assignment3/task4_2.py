import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim




import torchvision
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()  
        self.model = torchvision.models.resnet18(pretrained=True) 
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
                                             # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters 
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected 
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional 
            param.requires_grad = True # layers
    def forward(self, x):
        x = self.model(x)
        return x



def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    

    dataloaders = load_cifar10(batch_size)


  
    model = Model()
    model.to(utils.get_device())
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    create_plots(trainer, "task4_2")


if __name__ == "__main__":
    main()
