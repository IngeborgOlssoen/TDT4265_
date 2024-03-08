import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


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
        self.num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers


        # Initialize weights for Convolutional and Linear layers
        self.num_output_features = 128 * 4 * 4
        # Initialize our last fully connected layer
        
        
        
        self.feature_extractor = nn.Sequential(
            #Layer 1
            nn.Conv2d(in_channels=image_channels, out_channels=self.num_filters, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #Layer 2
            nn.Conv2d(in_channels=self.num_filters, out_channels=self.num_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.num_filters*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #Layer 3
            nn.Conv2d(in_channels=self.num_filters*2, out_channels=self.num_filters*4, kernel_size=5, padding=2),
            nn.BatchNorm2d(self.num_filters*4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        )
        
            
        
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
       

        #self.fc2= nn.Linear(in_features=image_channels, out_features= num_filters)
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        # Definer fully connected layers

        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    
        

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
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")


if __name__ == "__main__":
    main()
