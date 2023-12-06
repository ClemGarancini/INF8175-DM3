import nn
from backend import PerceptronDataset, RegressionDataset, DigitClassificationDataset


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
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(x,self.w)

    def get_prediction(self, x: nn.Constant) -> int:
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        y = nn.as_scalar(self.run(x))
        if y >= 0:
            prediction = 1
        else:
            prediction = -1
        return prediction
        

    def train(self, dataset: PerceptronDataset) -> None:
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        batch_size = 1
        all_correctly_classified = False
        while not all_correctly_classified:
            all_correctly_classified = True
            for x,y in dataset.iterate_once(batch_size):
                target = nn.as_scalar(y)
                prediction = self.get_prediction(x)
                if target != prediction:
                    all_correctly_classified = False
                    self.w.update(x,target)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self) -> None:
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.w = [nn.Parameter(1,20),nn.Parameter(20,5),nn.Parameter(5,1)]
        self.b = [nn.Parameter(1,20),nn.Parameter(1,5),nn.Parameter(1,1)]
        self.nb_couche_ffd = len(self.w)
        self.model = [nn.Linear, nn.AddBias,nn.ReLU,nn.Linear,nn.AddBias,nn.ReLU,nn.Linear,nn.AddBias]

    def run(self, x: nn.Constant) -> nn.Node:
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        nb_ffd = 0
        for layer in self.model:
            if layer == nn.Linear:
                x = layer(x, self.w[nb_ffd])
            if layer == nn.AddBias:
                x = layer(x, self.b[nb_ffd])
                nb_ffd +=1
            if layer == nn.ReLU:
                x = layer(x)
        return(x)

    def get_loss(self, x: nn.Constant, y: nn.Constant) -> nn.Node:
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        predicted_y = self.run(x)
        loss = nn. SquareLoss(predicted_y , y)
        return loss

    def train(self, dataset: RegressionDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        batch_size = 64
        step_size = -0.01
        size_dataset = dataset.x.shape[0]
        while size_dataset % batch_size !=0:
            batch_size = int(batch_size/2)
        loss = 1
        while loss > 0.02:
            for x,y in dataset.iterate_once(batch_size):
                step_loss = self.get_loss(x,y)
                nb_ffd = self.nb_couche_ffd

                # Liste contenant tous les paramètres
                parameters = self.w + self.b

                # Calcul des gradients
                grads = nn.gradients(step_loss, parameters)

                # Mise à jour des paramètres
                for i in range(nb_ffd):
                    self.w[i].update(grads[i], step_size)
                    self.b[i].update(grads[nb_ffd+i], step_size)
            for x,y in dataset.iterate_once(size_dataset):
                loss = nn.as_scalar(self.get_loss(x,y))
        



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
        self.w = [nn.Parameter(784,100),nn.Parameter(100,100),nn.Parameter(100,10)]
        self.b = [nn.Parameter(1,100),nn.Parameter(1,100),nn.Parameter(1,10)]
        self.nb_couche_ffd = len(self.w)
        self.model = [nn.Linear, nn.AddBias,nn.ReLU,nn.Linear,nn.AddBias,nn.ReLU,nn.Linear,nn.AddBias]

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
        nb_ffd = 0
        for layer in self.model:
            if layer == nn.Linear:
                x = layer(x, self.w[nb_ffd])
            if layer == nn.AddBias:
                x = layer(x, self.b[nb_ffd])
                nb_ffd +=1
            if layer == nn.ReLU:
                x = layer(x)
        return(x)

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
        y_pred = self.run(x)
        loss = nn.SoftmaxLoss(y_pred,y)
        return loss

    def train(self, dataset: DigitClassificationDataset) -> None:
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        batch_size = 25
        step_size = -0.07
        size_dataset = dataset.x.shape[0]
        while size_dataset % batch_size !=0:
            batch_size = int(batch_size/2)
        val_accuracy = 0
        while val_accuracy < 0.97:
            for x,y in dataset.iterate_once(batch_size):
                step_loss = self.get_loss(x,y)
                nb_ffd = self.nb_couche_ffd

                # Liste contenant tous les paramètres
                parameters = self.w + self.b

                # Calcul des gradients
                grads = nn.gradients(step_loss, parameters)

                # Mise à jour des paramètres
                for i in range(nb_ffd):
                    self.w[i].update(grads[i], step_size)
                    self.b[i].update(grads[nb_ffd+i], step_size)
            val_accuracy = dataset.get_validation_accuracy()
