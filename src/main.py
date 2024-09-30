from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST

from nn.activations.relu import ReLU
from nn.layers.linear import Linear
from nn.sequential.sequential import Sequential
from src.losses.categorical_cross_entropy import CategoricalCrossEntropy
from src.nn.nnelement import NNElement
from src.optimizers.adam import Adam

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the MNIST dataset and split it into training, validation, and testing sets.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training images,
    training labels, validation images, validation labels, testing images, and testing labels.
    """

    mndata = MNIST('../data')

    training_images, training_labels = mndata.load_training()

    validation_images = training_images[50000:]
    validation_labels = training_labels[50000:]

    training_images = training_images[:50000]
    training_labels = training_labels[:50000]

    testing_images, testing_labels = mndata.load_testing()

    return np.array(training_images), np.array(training_labels), np.array(validation_images), np.array(
        validation_labels), np.array(testing_images), np.array(testing_labels)


def create_network() -> NNElement:
    """
    Create a simple feedforward neural network with two hidden layers and ReLU activations.

    Returns:
    ------
    nn.sequential.Sequential: The neural network model.
    """

    optimizer = Adam(learning_rate=LEARNING_RATE)

    input_size = 784
    output_size = 10

    return Sequential(optimizer,
                      [
                          Linear(input_size, 32),
                          ReLU(),
                          Linear(32, 32),
                          ReLU(),
                          Linear(32, output_size),
                      ])


def get_batches(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    A generator that yields batches of images and labels of the specified batch size.
    It shuffles the data before creating the batches to ensure that the model sees different samples for optimality in
    stochastic gradient descent.

    Parameters
    ----------
    images (np.ndarray): The images to be batched.
    labels (np.ndarray): The labels to be batched.
    batch_size (int): The size of the batch.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]: A tuple containing the images and labels of the batch.
    """

    # Shuffle the data
    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        images = images[indices]
        labels = labels[indices]

    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size], labels[i:i + batch_size]


def plot(training_losses: list[np.floating], validation_losses: list[np.floating]) -> None:
    """
    Plot the training losses over time.

    Parameters
    ----------
    training_losses (list[float]): The training losses over time.
    validation_losses (list[float]): The validation losses over time.
    """

    plt.clf()

    plt.figure(figsize=(6, 6))

    plt.title('Training Loss Over Time')

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.plot(training_losses, label="Average Training Losses")
    plt.plot(validation_losses, label="Average Validation Losses")

    plt.legend()

    plt.pause(0.1)


def train_network(network: NNElement, training_images: np.ndarray, training_labels: np.ndarray,
                  validation_images: np.ndarray, validation_labels: np.ndarray) -> None:
    """
    Train the neural network on the training data.

    Parameters
    ----------
    network (NNElement): The neural network model.
    training_images (np.ndarray): The training images.
    training_labels (np.ndarray): The training labels.
    validation_images (np.ndarray): The validation images.
    validation_labels (np.ndarray): The validation labels
    """

    loss = CategoricalCrossEntropy(reduction='mean')

    training_losses = []
    validation_losses = []

    for epoch in range(EPOCHS):
        epoch_training_losses = []
        epoch_validation_losses = []

        for index, (batch_images, batch_labels) in enumerate(get_batches(training_images, training_labels, BATCH_SIZE)):
            print(f"\rTraining on batch {index} of epoch {epoch} ({len(batch_images)} samples)", end='', flush=True)

            y_pred = network.forward(batch_images)
            y_true = np.eye(y_pred.shape[1])[batch_labels]

            loss_value = loss.forward(y_pred, y_true)
            epoch_training_losses.append(loss_value)

            dloss_dpred = loss.backward()
            network.backward(dloss_dpred)

        print('')

        for index, (batch_images, batch_labels) in enumerate(
                get_batches(validation_images, validation_labels, BATCH_SIZE, shuffle=False)):
            print(f"\rValidation on batch {index} of epoch {epoch} ({len(batch_images)} samples)", end='', flush=True)

            y_pred = network.forward(batch_images)
            y_true = np.eye(y_pred.shape[1])[batch_labels]

            loss_value = loss.forward(y_pred, y_true)
            epoch_validation_losses.append(loss_value)

        print('')

        training_losses.append(np.mean(epoch_training_losses))
        validation_losses.append(np.mean(epoch_validation_losses))

        print(f"Epoch {epoch + 1}, Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}")
        plot(training_losses, validation_losses)


def test_network(network: NNElement, testing_images, testing_labels) -> None:
    """
    Test the neural network on the testing data.

    Parameters
    ----------
    network (NNElement): The neural network model.
    testing_images (np.ndarray): The testing images.
    testing_labels (np.ndarray): The testing labels.
    """

    correct = 0

    for index, (batch_images, batch_labels) in enumerate(get_batches(testing_images, testing_labels, BATCH_SIZE)):
        print(f"Testing on batch {index} ({len(batch_images)} samples)")

        predictions = network.forward(batch_images)

        correct += np.sum(np.argmax(predictions, axis=1) == batch_labels)

    print(f"Accuracy: {correct / len(testing_images)}")


def main():
    training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = load_data()

    network = create_network()

    train_network(network, training_images, training_labels, validation_images, validation_labels)
    test_network(network, testing_images, testing_labels)


if __name__ == "__main__":
    main()
