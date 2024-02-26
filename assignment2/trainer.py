import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images

np.random.seed(0)

############From assignment 1########
class BaseTrainer:

    def __init__(
            self,
            model,
            learning_rate: 0.02,
            batch_size: int,
            shuffle_dataset: bool,
            X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray, Y_val: np.ndarray,) -> None:
        """
            Initialize the trainer responsible for performing the gradient descent loop.
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = model
        self.shuffle_dataset = shuffle_dataset

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        pass

    def train_step(self):
        """
            Perform forward, backward and gradient descent step here.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        pass

    def train(
            self,
            num_epochs: int):
        """
        Training loop for model.
        Implements stochastic gradient descent with num_epochs passes over the train dataset.
        Returns:
            train_history: a dictionary containing loss and accuracy over all training steps
            val_history: a dictionary containing loss and accuracy over a selected set of steps
        """
        # Utility variables
        num_batches_per_epoch = self.X_train.shape[0] // self.batch_size
        num_steps_per_val = num_batches_per_epoch // 5
        # A tracking value of loss over all training steps
        train_history = dict(
            loss={},
            accuracy={}
        )
        val_history = dict(
            loss={},
            accuracy={}
        )

        global_step = 0
        best_val_loss=float("inf")
        val_loss_no_improve_count=0

        for epoch in range(num_epochs):
            train_loader = utils.batch_loader(
                self.X_train, self.Y_train, self.batch_size, shuffle=self.shuffle_dataset)
            for X_batch, Y_batch in iter(train_loader):
                loss = self.train_step(X_batch, Y_batch)
                # Track training loss continuously
                train_history["loss"][global_step] = loss

                # Track validation loss / accuracy every time we progress 20% through the dataset
                if global_step % num_steps_per_val == 0:
                    val_loss, accuracy_train, accuracy_val = self.validation_step()
                    train_history["accuracy"][global_step] = accuracy_train
                    val_history["loss"][global_step] = val_loss
                    val_history["accuracy"][global_step] = accuracy_val

                    # TODO (Task 2d): Implement early stopping here.
                    # You can access the validation loss in val_history["loss"]

                    #Early stopping check

                    if val_loss< best_val_loss:
                        best_val_loss=val_loss
                        val_loss_no_improve_count=0

                    else:
                        val_loss_no_improve_count+=1

                    if val_loss_no_improve_count>=10:
                        print(f"Stoping early at epoch {epoch} due to no improvement in validation loss.")

                        return train_history,val_history
                global_step += 1
        return train_history, val_history
    
########END OF COPY FROM ASSIGNMENT ONE###########


def calculate_accuracy(
    X: np.ndarray, targets: np.ndarray, model: SoftmaxModel
) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (copy from last assignment)
    
    output = model.forward(X)

    # Turn output into one hot encoded prediction
    max_indices = np.argmax(output, axis=1)
    pred = np.zeros_like(output)

    for id, row in zip(max_indices, pred):
        row[ id ] = 1
    
    N = X.shape[0]
    n_correct = 0 

    # Compare results
    for pred_row, target_row in zip(pred, targets):
        if np.array_equal(pred_row, target_row): n_correct += 1

    accuracy = n_correct / N

    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def __init__(
        self,
        momentum_gamma: float,
        use_momentum: 0.9,  # Task 3d hyperparmeter
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_gamma = momentum_gamma
        self.use_momentum = use_momentum
        # Init a history of previous gradients to use for implementing momentum
        self.previous_grads = [np.zeros_like(w) for w in self.model.ws]

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 2c)

        # Forward pass on all samples in batch
        output = self.model.forward(X_batch)
        
        # Backward pass to find gradient
        
        self.model.backward(X_batch, output, Y_batch)
        
        # Gradient update step
        self.model.ws[0] -= self.learning_rate * self.model.grads[0]
        self.model.ws[1] -= self.learning_rate * self.model.grads[1]
        
        # Add momentum term to update step
        if self.use_momentum:
            self.model.ws[0] -= self.learning_rate * self.momentum_gamma * self.previous_grads[0]
            self.model.ws[1] -= self.learning_rate * self.momentum_gamma * self.previous_grads[1]

            # Update previous gradients
            self.previous_grads[0] = self.model.grads[0]
            self.previous_grads[1] = self.model.grads[1]

        # Compute loss
        loss = cross_entropy_loss(Y_batch, output)
        return loss


    


    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits=self.model.forward(self.X_val)
        loss=cross_entropy_loss(self.Y_val, logits)

        accuracy_train=calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val=calculate_accuracy(self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs=50
    learning_rate=0.02 # Changed from 0.1 when using momentum
    batch_size=32
    neurons_per_layer=[64, 10]
    momentum_gamma=0.9  # Task 3 hyperparameter
    shuffle_data=True

    # Settings for task 2 and 3. Keep all to false for task 2.
    use_improved_sigmoid=True
    use_improved_weight_init=True
    use_momentum=True
    use_relu=False

    # Load dataset
    X_train, Y_train, X_val, Y_val=utils.load_full_mnist()
    X_train=pre_process_images(X_train)
    X_val=pre_process_images(X_val)
    Y_train=one_hot_encode(Y_train, 10)
    Y_val=one_hot_encode(Y_val, 10)
    # Hyperparameters

    model=SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )
    trainer=SoftmaxTrainer(
        momentum_gamma,
        use_momentum,
        model,
        learning_rate,
        batch_size,
        shuffle_data,
        X_train,
        Y_train,
        X_val,
        Y_val,
    )
    train_history, val_history=trainer.train(num_epochs)

    print(
        "Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model.forward(X_train)),
    )
    print(
        "Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model.forward(X_val)),
    )
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Plot loss for first model (task 2c)
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.9])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.99])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3?_train_loss.png")
    plt.show()


if __name__ == "__main__":
    main()