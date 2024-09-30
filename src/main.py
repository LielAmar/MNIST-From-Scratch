import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

from nn.activations.relu import ReLU
from nn.activations.softmax import Softmax
from nn.layers.linear import Linear
from nn.sequential.sequential import Sequential
from src.constants import EPOCHS, BATCH_SIZE
from src.nn.layers.layer import Layer
from src.nn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from src.nn.optimizers.adam import Adam


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mndata = MNIST('../data')

    training_images, training_labels = mndata.load_training()

    validation_images = training_images[50000:]
    validation_labels = training_labels[50000:]

    training_images = training_images[:50000]
    training_labels = training_labels[:50000]

    testing_images, testing_labels = mndata.load_testing()

    return np.array(training_images), np.array(training_labels), np.array(validation_images), np.array(
        validation_labels), np.array(testing_images), np.array(testing_labels)


def create_network() -> Sequential:
    input_size = 784
    output_size = 10

    hidden_layer_1 = 64
    hidden_layer_2 = 64

    return Sequential([
        Linear(input_size, hidden_layer_1),
        ReLU(),
        Linear(hidden_layer_1, hidden_layer_2),
        ReLU(),
        Linear(hidden_layer_2, output_size),
        Softmax()
    ])


def get_batches(images, labels, batch_size) -> tuple[np.ndarray, np.ndarray]:
    # Shuffle the data
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    for i in range(0, len(images), batch_size):
        yield images[i:i + batch_size], labels[i:i + batch_size]


def update_plot(losses):
    plt.clf()

    plt.figure(figsize=(6, 6))

    plt.title('Training Loss Over Time')

    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.plot(losses, label="Average Losses")

    plt.legend()

    plt.pause(0.1)


def train_network(network: Sequential, training_images, training_labels) -> None:
    optimizer = Adam(learning_rate=0.0005)
    loss = CategoricalCrossEntropy(reduction='mean')

    # Initialize optimizer for each layer with weights
    for layer in network.sequence:
        if isinstance(layer, Layer):
            optimizer.initialize(layer)

    losses = []

    for epoch in range(EPOCHS):
        epoch_losses = []

        for index, (batch_images, batch_labels) in enumerate(get_batches(training_images, training_labels, BATCH_SIZE)):
            print(f"\rTraining on batch {index} of epoch {epoch} ({len(batch_images)} samples)", end='', flush=True)

            y_pred = network.forward(batch_images)
            y_true = np.eye(y_pred.shape[1])[batch_labels]

            raw_loss = loss.forward(y_pred, y_true)

            dL_dpred = loss.backward()

            dL_dW, dL_db = None, None

            for nn_element in reversed(network.sequence):
                if isinstance(nn_element, Layer):
                    dL_dpred, dL_dW, dL_db = nn_element.backward(dL_dpred)

                    optimizer.update(nn_element, dL_dW, dL_db)
                else:
                    dL_dpred = nn_element.backward(dL_dpred)

            epoch_losses.append(raw_loss)

        losses.append(np.mean(epoch_losses))
        update_plot(losses)

        # Debugging: Print epoch loss
        print(f"Epoch {epoch + 1}, Loss: {losses[-1]}")


def test_network(network: Sequential, testing_images, testing_labels) -> None:
    correct = 0

    for index, (batch_images, batch_labels) in enumerate(get_batches(testing_images, testing_labels, BATCH_SIZE)):
        print(f"Testing on batch {index} ({len(batch_images)} samples)")

        predictions = network.forward(batch_images)

        correct += np.sum(np.argmax(predictions, axis=1) == batch_labels)

    print(f"Accuracy: {correct / len(testing_images)}")


def main():
    training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = load()

    network = create_network()

    train_network(network, training_images, training_labels)
    test_network(network, testing_images, testing_labels)


if __name__ == "__main__":
    main()
