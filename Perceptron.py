"""
A simple implementation of a linear separator.
Inspired by lectures and slides by Ray Mooney, University of Texas at Austin,
describing the original Perceptron developed by Frank Rosenblatt in 1957.

Written by Ethan Mason.
"""
class Perceptron:
    """
    Initialize the weight vector the Perceptron will use, self.weights,
    along with the threshold and learning rate.

    params:
        initWeight:     the value all weights will start with
        k:              the length of the vectors used in the given dataset 
        threshold:      the threshold to be used in training
        learningRate:   the learning rate to be used in training
    """
    def __init__(self, initWeight, k, threshold, learningRate):
        assert k > 0 and learningRate > 0
        self.weights = [initWeight] * k
        self.threshold = threshold
        self.learningRate = learningRate
    
    """
    Adjusts the weights in the event of an incorrect classification.

    If output is high, lower weights on active inputs.
    If output is low, increase weights on active inputs.
    If output is correct, the weights won't change.
    """
    def updateWeights(self, exampleWeights, output, teacherOutput):
        assert len(self.weights) == len(exampleWeights)
        for i in range(len(self.weights)):
            self.weights[i] += self.learningRate * (teacherOutput - output) * exampleWeights[i]

    """
    Adjusts the threshold in the event of an incorrect classification.

    If there was a false positive, increase threshold.
    If there was a false negative, decrease threshold.
    In the case of a correct classification, the threshold is unchanged.
    """
    def updateThreshold(self, output, teacherOutput):
        self.threshold -= self.learningRate * (teacherOutput - output)

    """
    Classifies the training example using the current weight vector.
    If the dot product of the weight vector and the example
    is greater than or equal to the threshold, return True.
    Return false otherwise.

    Assumes example is a k-tuple representing a weight vector.
    """
    def classify(self, example):
        assert len(self.weights) == len(example)
        dotProduct = 0
        for i in range(len(self.weights)):
            dotProduct += self.weights[i] * example[i]
        
        return dotProduct >= self.threshold

    """
    Returns True if the Perceptron classfies all training examples correctly,
    False otherwise.

    Assumes examples is a list of 2-tuples, of the form:

    ((input weights), classification)

    Where the first item is a weight vector represented as a k-tuple,
    and the second item is the classification modeled as True or False.
    """
    def allCorrect(self, examples):
        for i in range(len(examples)):
            correctClassification = examples[i][1]
            # Classify the example using only the first item, the weight vector
            if self.classify(examples[i][0]) != correctClassification:
                return False
        return True

    """
    Iteratively update weights until convergence 
    (i.e. outputs of all training example are correct)
    or 1000 epochs have passed. This limit is in place
    for datasets that aren't linearly separable.

    Assumes examples is a list of tuples, of the form:

    ((input weights), classification)

    Where the first item is a weight vector represented as a k-tuple,
    and the second item is the classification modeled as True or False.
    """
    def train(self, examples):
        print(f"Initial Weights: {self.weights}")
        print(f"Initial Threshold: {self.threshold}\n")
        epoch = 1
        allCorrect = self.allCorrect(examples)
        # Safeguard the loop in case the weights begin thrashing
        while not allCorrect and epoch < 1000:
            print(f"Epoch {epoch}:")
            print(f"Weights: {self.weights}")
            print(f"Threshold: {self.threshold}\n")
            for e in examples:
                # Get training example info
                exampleWeights = e[0]
                teacherOutput = e[1]
                print(f"Example {exampleWeights}")
                print(f"Expected output: {teacherOutput}")
                # Compute current output for the example
                output = self.classify(exampleWeights)
                print(f"Output: {output}\n")
                if output != teacherOutput:
                    # Update weights and threshold using Perceptron learning rule
                    print("Incorrect - updating weights")
                    self.updateWeights(exampleWeights, output, teacherOutput)
                    self.updateThreshold(output, teacherOutput)
                else:
                    print("Correct!")
                print(f"Weights: {self.weights}")
                print(f"Threshold: {self.threshold}\n")
            epoch += 1
            allCorrect = self.allCorrect(examples)

        print("Training done.")
        print(f"Weights: {self.weights}")
        print(f"Threshold: {self.threshold}\n")
        for e in examples:
            print(f"Example {e}")
            print(f"Output: {self.classify(e[0])}")
        
        if (allCorrect):
            print("\nAll correct!")
        else:
            print("\nNot all correct, something went wrong!")
        

if __name__ == "__main__":
    exampleOne = ((0, 1, 0), True)
    exampleTwo = ((1, 1, 0), True)
    exampleThree = ((1, 0, 0), True)
    exampleFour = ((0, 0, 0), True)
    exampleFive = ((0, 0, 1), False)
    exampleSix = ((1, 1, 1), False)
    examples = (exampleOne, exampleTwo, exampleThree, exampleFour, exampleFive, exampleSix)
    # Create a simple Perceptron with initial weights of 0, examples of length 3,
    # initial threshold of 0, and learning rate of 1
    p = Perceptron(0, 3, 0, 1)
    p.train(examples)

    exampleSeven = ((1, 0, 1), False)
    print(f"\nPost-training example: {exampleSeven}")
    output = p.classify(exampleSeven[0])
    print(f"Output: {output}")
    if (output == exampleSeven[1]):
        print("Correct!")
    else:
        print("Incorrect.")
