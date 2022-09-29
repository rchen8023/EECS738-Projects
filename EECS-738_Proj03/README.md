# EECS-738_Proj03

In this project, I build a Neural Network model to do classification prediction in two datasets

# Project Instruction

Neural network architectures  

1. Set up a new git repository in your GitHub account  
2. Pick two datasets from  
https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research  
3. Choose a programming language (Python, C/C++, Java)  
4. Formulate ideas on how neural networks can be used to accomplish the task for the specific dataset  
5. Build a neural network to model the prediction process programmatically  
6. Document your process and results  
7. Commit your source code, documentation and other supporting files to the git repository in GitHub  

# Dataset

The first dataset is CIFAR from PyTorch, which contains 60000 image datas in 10 classes:  
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html  
The dataset for CIFAR is too large, so I cannot update to github file, but the python code can directly download and load the dataset

The second dataset is Ablone dataset from UCI Machine Learning Repository, which contains 4177 instances in 3 classes:  
https://archive.ics.uci.edu/ml/datasets/Abalone  

# Model

**Input Layer:**

The input layer is all instances, and number of nodes in this layer is the number of features of the dataset

**Hidden Layer:**

This is a simple model that only contains one hidden layer, and the number of nodes in the hidden layer can be modified as input parameter, in my processing, I used 20 nodes.

**Output Layer:**

The number of nodes in output layer is the number of classes in the dataset.

**Activation Function:**

I listed three possible activation function: sigmoid, relu, and tanh. For two models, one model I used the same activation function for both hidden layer and output layer, and the type of activation function can be modified as input. The second model I used different activation function for hidden layer and output layer, the function on output layer is fixed to softmax, and for hidden layer, the function can be modified as input. 


# Results

As result, both model can reduce the cost values (cross entropy and mean square error). But the accuracy of both model are not very good. I got 54% for Abalone dataset, and 10.9% for CIFAR dataset. I think that may because both of my dataset are contain more than one classes, and one of the data contains large number of features, so for this kind of dataset, we may need a more complex model with more number of nodes in hidden layer and more hidden layers may increase the accuracy. 


# References
https://realpython.com/python-ai-neural-network/  
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6  
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/  
https://pytorch.org/vision/stable/datasets.html#cifar  
https://pytorch.org/text/stable/datasets.html#amazonreviewfull  
https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/  
