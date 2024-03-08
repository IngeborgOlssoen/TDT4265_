import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from torchvision import transforms
from trainer import Trainer, compute_loss_and_accuracy
from torch.optim.lr_scheduler import StepLR

class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        self.num_classes = num_classes
        kernel_size_conv = 3
        kernel_size_else = 2
        # Define the convolutional layers


        # Initialize weights for Convolutional and Linear layers
        self.num_output_features =  2 * 2 * 1024
        # Initialize our last fully connected layer
        
        
        
        self.feature_extractor = nn.Sequential(
            #Layer 1
            nn.Conv2d(in_channels=image_channels, out_channels=128, kernel_size=kernel_size_conv, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=kernel_size_else, stride=2),
            
            #Layer 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size_conv, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=kernel_size_else, stride=2),
            
            #Layer 3
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size_conv, stride= 1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=kernel_size_else, stride=2),

            #Layer 4
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=kernel_size_conv, stride= 1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=kernel_size_else, stride=2),

        )
        
            
        
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
       

        #self.fc2= nn.Linear(in_features=image_channels, out_features= num_filters)
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        # Definer fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),

            #Layer 5
            nn.Linear(self.num_output_features, 512),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.05),

            #Layer 6
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.05),

            #Layer 7
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        
        # Pass input through the feature extractor layers
        x= self.feature_extractor(x)
        x=x.view(-1,self.num_output_features)
        x= self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

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

def load_cifar10(batch_size, augment=False):
    # Define the standard normalization for CIFAR-10
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.25, 0.25, 0.25])
    
    # When augmenting, add random transformations to the training set
    transform_list = [transforms.ToTensor(), normalize]
    if augment:
        transform_list = [transforms.RandomHorizontalFlip(),
                          transforms.RandomCrop(32, padding=4)] + transform_list
    
    transform_train = transforms.Compose(transform_list)
    
    # The validation and test sets should not be augmented, only normalized
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Download and load the training data with augmentations
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Split trainset to create a validation set
    val_size = 5000
    train_size = len(trainset) - val_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Use the non-augmented transformations for validation and test sets
    val_dataset.dataset.transform = transform_test
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader





def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4

    # Load the CIFAR10 data
    train_loader, val_loader, test_loader = load_cifar10(batch_size=64, augment=True)
    dataloaders = [train_loader, val_loader, test_loader]

    # Create the model
    model = ExampleModel(image_channels=3, num_classes=10)

    # Create the trainer instance and pass the list of data loaders
    trainer_instance = Trainer(
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stop_count=early_stop_count,
        epochs=epochs,
        model=model,
        dataloaders=dataloaders, # Pass the list of data loaders
        opt="Adam",
        weight_decay=0.0,
        momentum=0.6
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    trainer_instance.optimizer = optimizer  # Assigning optimizer after initialization

    for epoch in range(epochs):
        trainer_instance.train()  
        val_loss, val_acc = trainer_instance.validate()
        scheduler.step()   
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"Training Loss: {trainer_instance.train_history['loss'][-1]}")
        print(f"Validation Loss: {trainer_instance.validation_history['loss'][-1]}")
        print(f"Validation Accuracy: {trainer_instance.validation_history['accuracy'][-1]}")

    create_plots(trainer_instance, "task3")

    train_loss, train_acc = compute_loss_and_accuracy(
        trainer_instance.dataloader_train, trainer_instance.model, trainer_instance.loss_criterion
    )
    print("Train loss: ", train_loss)
    print("Train accuracy: ", train_acc)

    val_loss, val_acc = compute_loss_and_accuracy(
        trainer_instance.dataloader_val, trainer_instance.model, trainer_instance.loss_criterion
    )
    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc)

    test_loss, test_acc = compute_loss_and_accuracy(
        trainer_instance.dataloader_test, trainer_instance.model, trainer_instance.loss_criterion
    )
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_acc)

if __name__ == "__main__":
    main()
    val_loss, val_acc = trainer_instance.validate()
    print(f"Final Validation Loss: {val_loss}, Final Validation Accuracy: {val_acc}")
