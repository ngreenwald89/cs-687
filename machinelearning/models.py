import nn
from time import time

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # print(f'x: {x}')
        # print(f'self.run(x): {self.run(x)}')
        # print(f'nn.as_scalar(self.run(x)): {nn.as_scalar(self.run(x))}')
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        '''Write the train(self) method. This should repeatedly loop over the data set and make updates on examples that are misclassified. 
        Use the update method of the nn.Parameter class to update the weights. 
        When an entire pass over the data set is completed without making any mistakes, 
        100% training accuracy has been achieved, and training can terminate.
In this project, the only way to change the value of a parameter is by calling parameter.update(direction, multiplier), 
which will perform the update to the weights:
    weights ← weights +direction ⋅ multiplier
    The direction argument is a Node with the same shape as the parameter, and the multiplier argument is a Python scalar.'''

        "*** YOUR CODE HERE ***"
        updated = True  # initialize updated to True to start while loop
        i = 0
        while updated:
            updates = 0
            updated = False  # keep training until no updates, meaning 100% accuracy
            for x, y in dataset.iterate_once(1):
                y_as_scalar = nn.as_scalar(y)
                p = self.get_prediction(x)
                if p > y_as_scalar:
                    self.w.update(x, -1)
                    updates += 1
                    updated = True
                elif p < y_as_scalar:
                    self.w.update(x, 1)
                    updates += 1
                    updated = True
            print(f'i: {i}, updates: {updates}')
            i += 1


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 13  # recommended between 10 and 400
        self.batch_size = 20  # between 1 and size of dataset. Require total size of dataset evenly divisible by batch_size.  q2 dataset size=200
        self.learning_rate = .04  # between .001 and 1.0
        self.num_hidden_layers = 1  # between 1 and 3
        self.w1 = nn.Parameter(1, self.hidden_layer_size)  # input for q2 has one feature
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, 1)  # output for q2 has one feature
        self.b2 = nn.Parameter(1, 1)
        self.start = time()

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x, self.w1)
        # print(f'xm: {xm}, b: {self.b1}')
        activation_1 = nn.ReLU(nn.AddBias(xm, self.b1))  # shape: (batch_size x hidden_layer_size)
        # print(f'activation_1: {activation_1}')
        activation_2 = nn.Linear(activation_1, self.w2)  # shape: (hidden_layer_size x output_dim)
        # print(f'activation_2: {activation_2}')
        predicted_y = nn.AddBias(activation_2, self.b2)
        # print(f'predicted_y: {predicted_y.data}')
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SquareLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        i = 0
        while True:
            cumulative_loss = 0
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # print(f'loss: {loss.data}')
                grad_wrt_w2, grad_wrt_b2, grad_wrt_w1, grad_wrt_b1 = nn.gradients(loss, [self.w2, self.b2, self.w1, self.b1])
                self.w2.update(grad_wrt_w2, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
                # print(f'grad_wrt_2.data: {grad_wrt_w2.data}')

                self.w1.update(grad_wrt_w1, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)

                # print(f'grad_wrt_w1: {grad_wrt_w1}, self.w1: {self.w1}')

                cumulative_loss += nn.as_scalar(loss)
            if i % 100 == 0:
                time_elapsed = round(time() - self.start, 2)
                print(f'i: {i}, cumulative_loss: {cumulative_loss}: time elapsed: {time_elapsed} seconds')
            i += 1
            if cumulative_loss <= .02:
                print(f'Success! hidden_layer_size: {self.hidden_layer_size}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}')
                return


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
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_layer_size = 100  # recommended between 10 and 400
        self.batch_size = 25  # between 1 and size of dataset. Require total size of dataset evenly divisible by batch_size.  q2 dataset size=200
        self.learning_rate = .05  # between .001 and 1.0
        self.num_hidden_layers = 1  # between 1 and 3
        self.w1 = nn.Parameter(784, self.hidden_layer_size)  # input for q2 has one feature
        self.b1 = nn.Parameter(1, self.hidden_layer_size)
        self.w2 = nn.Parameter(self.hidden_layer_size, 10)  # output for q2 has one feature
        self.b2 = nn.Parameter(1, 10)
        self.start = time()

    def run(self, x):
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
        "*** YOUR CODE HERE ***"
        xm = nn.Linear(x, self.w1)
        activation_1 = nn.ReLU(nn.AddBias(xm, self.b1))  # shape: (batch_size x hidden_layer_size)
        activation_2 = nn.Linear(activation_1, self.w2)  # shape: (hidden_layer_size x output_dim)
        predicted_y = nn.AddBias(activation_2, self.b2)
        return predicted_y

    def get_loss(self, x, y):
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
        "*** YOUR CODE HERE ***"
        predicted_y = self.run(x)
        return nn.SoftmaxLoss(predicted_y, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        print(
            f'Training Parameters - hidden_layer_size: {self.hidden_layer_size}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}')
        i = 0
        while True:

            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w2, grad_wrt_b2, grad_wrt_w1, grad_wrt_b1 = nn.gradients(loss,
                                                                                  [self.w2, self.b2, self.w1, self.b1])
                self.w2.update(grad_wrt_w2, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)

                self.w1.update(grad_wrt_w1, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)

            time_elapsed = round(time() - self.start, 2)
            validation_accuracy = dataset.get_validation_accuracy()
            print(f'i: {i}, validation_accuracy: {validation_accuracy}: time elapsed: {time_elapsed} seconds')
            i += 1
            if validation_accuracy >= .9705:
                print(
                    f'Success! hidden_layer_size: {self.hidden_layer_size}, batch_size: {self.batch_size}, learning_rate: {self.learning_rate}')
                return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
