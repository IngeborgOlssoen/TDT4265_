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


from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (.25, .25, .25))
    ])
    # Last inn hele datasettet
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Del datasettet i trening og validering
    train_set, val_set = random_split(dataset, [45000, 5000])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Last inn testdatasettet
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    

    train_loader, val_loader, test_loader = load_cifar10(batch_size)


  
    model = Model().to(utils.get_device())
    
    trainer = Trainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        momentum=0.9,  # Antar at dette er momentum-verdien du vil bruke
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model,
        dataloaders=[train_loader, val_loader, test_loader],  # Viktig: Passer datalasterne som en liste
        opt="Adam",
        weight_decay=0.0  # Antar at dette er weight_decay-verdien du vil bruke
    )
    
    trainer.train()
    create_plots(trainer, "task4_2")


if __name__ == "__main__":
    main()
