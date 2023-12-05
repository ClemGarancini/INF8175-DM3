import nn
import numpy as np
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset
import matplotlib.pyplot as plt


class PerceptronModel(object):
    def __init__(self, dimensions: int) -> None:
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self) -> nn.Parameter:
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        # Compute the dot product
        x_dot_w = nn.DotProduct(x, self.w)
        # Create the Node
        y = nn.Constant(x_dot_w.data)
        return y

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        # Compute dot product
        dot_product = self.run(x)
        # Convert to scalar
        dot_product_scalar = nn.as_scalar(dot_product)
        # Create y Node
        if dot_product_scalar == 0:
            y = nn.Constant(np.array(1.0))
        else:
            y = nn.Constant(np.array(np.sign(dot_product_scalar)))
        return nn.as_scalar(y)

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        DEBUG = False
        badClassification = True
        while badClassification:
            badClassification = False
            precision = 0
            for x, y in dataset.iterate_once(1):
                y_pred = self.get_prediction(x)
                if DEBUG:
                    print(
                        "x is {}, y is {}, y_pred is {}".format(
                            x, nn.as_scalar(y), y_pred
                        )
                    )
                badClassification = badClassification or (y_pred != nn.as_scalar(y))
                if y_pred != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    precision += 1
            if DEBUG:
                print(badClassification)
                print(
                    "An episode has ended, the precision is: {}%".format(
                        100 * (dataset.x.shape[0] - precision) / dataset.x.shape[0]
                    )
                )

        return None


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        self.batch_size = 25
        self.nbParametersLayer1 = 100
        self.learning_rate = -0.07

        # Model architecture
        # Hidden layer:
        ### Weights
        self.layer1Parameters = nn.Parameter(1, self.nbParametersLayer1)
        ### Bias
        self.layer1Bias = nn.Parameter(1, self.nbParametersLayer1)

        # Output layer:
        ### Weights
        self.layer2Parameters = nn.Parameter(self.nbParametersLayer1, 1)
        ### Bias
        self.layer2Bias = nn.Parameter(1, 1)

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Forward Pass of the 1st hidden layer
        a = nn.ReLU(nn.AddBias(nn.Linear(x, self.layer1Parameters), self.layer1Bias))

        # Forward Pass of the output layer
        y_pred = nn.AddBias(nn.Linear(a, self.layer2Parameters), self.layer2Bias)

        return y_pred

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        y_pred = self.run(x)
        loss = nn.SquareLoss(y, y_pred)
        return loss

    def customLoss(self, x: nn.Constant, y: nn.Constant) -> float:
        l = 0
        for i in range(len(x.data)):
            l += 0.5 * (x.data[i][0] - y.data[i][0]) ** 2
        return l / len(x.data)

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        DEBUG = True
        done = False
        e = 0
        while not done:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(
                    loss,
                    [
                        self.layer1Parameters,
                        self.layer1Bias,
                        self.layer2Parameters,
                        self.layer2Bias,
                    ],
                )
                self.layer1Parameters.update(grad_w1, self.learning_rate)
                self.layer1Bias.update(grad_b1, self.learning_rate)
                self.layer2Parameters.update(grad_w2, self.learning_rate)
                self.layer2Bias.update(grad_b2, self.learning_rate)

            # print(self.customLoss(nn.Constant(dataset.y), y_pred))
            # print(nn.Constant(dataset.y), y_pred)
            # print("loss")
            loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            done = loss.data.item() < 0.02
            e += 1
            if DEBUG:
                print("Episode {} ended, loss is {}".format(e, loss.data.item()))
                # wTest = nn.Parameter(1, 1)
                # xTest = nn.Constant(np.array([[1]], dtype=np.float32))
                # yPredTest = nn.Linear(xTest, wTest)
                # yTest = nn.Constant(np.array([[4]], dtype=np.float32))
                # print("###########################")
                # print("y: ", yTest.data.item(), "\nyPred: ", yPredTest.data.item())

                # lossTest = nn.SquareLoss(yPredTest, yTest)
                # print("loss: ", lossTest.data.item())

                # print("parameters where: ", wTest.data)
                # [grad] = nn.gradients(lossTest, [wTest])
                # print("grad is: ", grad.data)
                # wTest.update(grad, 1)
                # print("parameters are: ", wTest.data)
                # plt.figure()
                # plt.plot(dataset.x, y_pred.data)
                # plt.show()
        print(
            "Training has ended after {} episodes, final loss is {}".format(
                e, loss.data.item()
            )
        )


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
