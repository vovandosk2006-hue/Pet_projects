import numpy as np
import matplotlib.pyplot as plt

# open train and test files for binary reading
train_images = open("train-images.idx3-ubyte", "rb")
train_labels = open("train-labels.idx1-ubyte", "rb")
test_images = open("t10k-images.idx3-ubyte", "rb")
test_labels = open("t10k-labels.idx1-ubyte", "rb")


# read info from train_images stream normalize images data
magic = int.from_bytes(train_images.read(4), "big")
num_images = int.from_bytes(train_images.read(4), "big")
rows = int.from_bytes(train_images.read(4), "big")
cols = int.from_bytes(train_images.read(4), "big")

images = np.frombuffer(train_images.read(), dtype = np.uint8)
images = images.reshape(num_images, rows, cols)
images = images / 255

# read info from train_labels stream
magic2 = int.from_bytes(train_labels.read(4), "big")
size = int.from_bytes(train_labels.read(4), "big")
labels = np.frombuffer(train_labels.read(), dtype = np.uint8)

# read info from test_images stream and normalize images data
magic3 = int.from_bytes(test_images.read(4), "big")
num_images2 = int.from_bytes(test_images.read(4), "big")
rows2 = int.from_bytes(test_images.read(4), "big")
cols2 = int.from_bytes(test_images.read(4), "big")

images2 = np.frombuffer(test_images.read(), dtype = np.uint8)
images2 = images2.reshape(num_images2, rows2, cols2)
images2 = images2 / 255

# read info from test_labels stream
magic4 = int.from_bytes(test_labels.read(4), "big")
size2 = int.from_bytes(test_labels.read(4), "big")
labels2 = np.frombuffer(test_labels.read(), dtype = np.uint8)

# initialize layers sizes(count of neurones)
input_size = 784
output_size = 10
hidden_size = 20

# create the activate function and derivate it
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# create random weights and zero biases
weights1 = np.random.uniform(-1, 1, (input_size, hidden_size))
bias1 = np.zeros(hidden_size)
weights2 = np.random.uniform(-1, 1, (hidden_size, output_size))
bias2 = np.zeros(output_size)

# create epochs counter, learning_rate and 2 lists which keeps train and test losses
epochs = 20
learning_rate = 0.1
train_loss = []
test_loss = []

for e in range(epochs):
    train_epoch_loss = 0
    test_epoch_loss = 0
    print(f"epoch № {e + 1}/{epochs}:")

    # model training
    for image, label in zip(images, labels):
        image_vector = image.flatten()
        lable_vector = np.zeros(output_size)
        lable_vector[label] = 1

        # forward propagation

        inputs_output = image_vector @ weights1 + bias1

        hidden = sigmoid(inputs_output)
        hidden_output = hidden @ weights2 + bias2

        output = sigmoid(hidden_output)

        MSE = 1/output_size * np.sum((output - lable_vector) ** 2)

        train_epoch_loss += MSE

        # backpropagation

        derivate_MSE_for_output = 2 * (output - lable_vector) / output_size
        derivate_output_for_hidden_output = derivative_sigmoid(hidden_output)
        derivate_hidden_output_for_weights2 = hidden
        derivate_hidden_output_for_bias2 = derivate_inputs_output_for_bias1 = 1

        derivate_MSE_for_weights2 = np.outer(derivate_hidden_output_for_weights2, (derivate_MSE_for_output * derivate_output_for_hidden_output))
        derivate_MSE_for_bias2 = derivate_MSE_for_output * derivate_output_for_hidden_output * derivate_hidden_output_for_bias2

        derivate_hidden_output_for_hidden = weights2
        derivate_hidden_for_inputs_output = derivative_sigmoid(inputs_output)
        derivate_inputs_output_for_weights1 = image_vector

        derivate_MSE_for_weights1 = np.outer(derivate_inputs_output_for_weights1, (derivate_MSE_for_output * derivate_output_for_hidden_output @ derivate_hidden_output_for_hidden.T * derivate_hidden_for_inputs_output))
        derivate_MSE_for_bias1 = derivate_MSE_for_output * derivate_output_for_hidden_output @ derivate_hidden_output_for_hidden.T * derivate_hidden_for_inputs_output * derivate_inputs_output_for_bias1

        weights2 -= learning_rate * derivate_MSE_for_weights2
        bias2 -= learning_rate * derivate_MSE_for_bias2
        weights1 -= learning_rate * derivate_MSE_for_weights1
        bias1 -= learning_rate * derivate_MSE_for_bias1

    # model testing
    for image, label in zip(images2, labels2):
        image_vector = image.flatten()
        lable_vector = np.zeros(output_size)
        lable_vector[label] = 1

        inputs_output = image_vector @ weights1 + bias1

        hidden = sigmoid(inputs_output)
        hidden_output = hidden @ weights2 + bias2

        output = sigmoid(hidden_output)

        MSE = 1/output_size * np.sum((output - lable_vector) ** 2)
        test_epoch_loss += MSE

    train_epoch_loss = train_epoch_loss / len(images)
    train_loss.append(train_epoch_loss)
    print(f"train_epoch_loss = {train_epoch_loss}")

    test_epoch_loss = test_epoch_loss / len(images2)
    test_loss.append(test_epoch_loss)
    print(f"test_epoch_loss = {test_epoch_loss}")

# plot train-test loss graph
plt.figure(figsize=(8,5))
plt.plot(train_loss, label="Train Loss", color="blue")
plt.plot(test_loss, label="Test Loss", color="red")
plt.title("Train vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()