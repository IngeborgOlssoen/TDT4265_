import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


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

        #for m in self.modules():
         #   if isinstance(m, nn.Conv2d):
          #      nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
           #     if m.bias is not None:
            #        nn.init.constant_(m.bias, 0)
            #elif isinstance(m, nn.Linear):
             #   nn.init.normal_(m.weight, 0, 0.01)
              #  nn.init.constant_(m.bias, 0)

    
        #self.fc1 = nn.Linear(in_features=image_channels,out_features=self.num_filters)
    
        

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        
        print(batch_size)
        # Pass input through the feature extractor layers

        x= self.feature_extractor(x)
        print(x)
        print(x.size())
        
        x=x.view(-1,self.num_output_features)
        print(x.size())
        x= self.classifier(x)
        print(x.size())
        

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
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task3")

    
    train_loss, train_acc = compute_loss_and_accuracy(
        trainer.dataloader_train, trainer.model, trainer.loss_criterion
    )
    print("Train loss: ", train_loss)
    print("Train accuracy: ", train_acc)

    val_loss, val_acc = compute_loss_and_accuracy(
        trainer.dataloader_val, trainer.model, trainer.loss_criterion
    )
    print("Validation loss: ", val_loss)
    print("Validation accuracy: ", val_acc)

    test_loss, test_acc = compute_loss_and_accuracy(
        trainer.dataloader_test, trainer.model, trainer.loss_criterion
    )
    print("Test loss: ", test_loss)
    print("Test accuracy: ", test_acc)


if __name__ == "__main__":
    main()
