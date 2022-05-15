# Project_I: Neural Networks

This Python script can initialise a neural network object of any size. You can then get attributes of the object, generate outputs, and improve the network using backpropagation.

I started this project after watching 3Blue1Brown's series on the Neural Networks (https://www.3blue1brown.com/topics/neural-networks). He is excellent at explaining the logic behind it, which made deriving the formulas a lot easier.

This is primarily a learning project for me; I went into this project completely ignorant of what a Neural Network is and I wanted to see how far I could get from just knowing the basic principles behind back-propagation and using formulas I had derived. 
There are definitely more sophisticated ways to do it, but I hope anyone using this can gain some insight from a complete novice's interpretation of Neural Networks. 


## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Usage](#usage)

## General info

There are 3 important files:

##### DerivationBackpropagation.pdf

This is a pdf of the formulas that I've used to create this program (the previous notes look a lot scruffier!).
Feel free to point out any mistakes I've made; the only way I know that this is correct is because it works, so help is always appreciated if you see something! 

#### Inside Project_I directory
##### Mathssss.py

Nothing wrong with a few extra s'. 
I'm sure that numpy would have done the trick here, but who doesn't love a sprinkling of extra complexity.
This is a file with all of the mathematical functions that I may have needed at some point during this project; and yes some of the functions could be named better...

##### NeuralNetworks_2.py

Neuralnetworks.py was my original attempt. I got fairly far before realising that I needed to take the notation seriously! But I have left it in because I thought it could be educational.

NN_2 is where the magic happens! 

Upon initialisation, the program will generate the structure of all of the attributes inserting placeholders – or random values in the case of weights and biases. The program will generate serious values for these later.

forward_propagation(): takes in an input, and an expected output (y). It will then calculate the values of all of the nodes, including the output, and calculate the c_by_last_layer which is formula 1 in my derivation sheet.

The gen_cbprop_matrices() method calculates formula 2 for us. The name stands for "cumulative backpropagation matrices" which is the multiplicative sum of all of the backpropagation matrices on each layer. This was a toughy! It required a lot of careful attention to the indices.

gen_cost_by_weight() and gen_cost_by_bias() do as they say on the tin. Given a layer of the network and the relevant indices to tell the program where the weight or bias is, the network will return the cost of that particular weight or bias. 

adjust_weights_and_bias() puts all of this together and adjusts the network's weights and biases depending on the cost to the network for the most recent output. (This needs changing so that it adjusts the cost for lots of inputs rather than just the one)

## Technologies
Project created with:
 - Python 3.10.0
 - numpy 1.21.4

## Usage

#### Step 1 - Initialise a NeuralNetwork
Network = NeuralNetwork( [layers] )
the init function takes in one argument, a list of the number of nodes in each layer. 

e.g. NeuralNetwork( [4,5,6,5,4] ) is a network with 5 layers with the nodes on each layer corresponding to the values of the list respectively.

#### Step 2 - Feeding in data

The way that i've been using this class is by creating a pandas dataframe using data from a csv file.
For each set of training data that you want to train your network on, you'll need to loop over each input and feed it into the forward_propagation() function.
Then do the gen_cost_gradient() (I could probably put this inside the forward_propagation function, but it's useful to keep it out sometimes).
A loop can look something like this:

for case in test_cases:
    forward_propagation(input,expected_output)
    gen_cost_gradient()

#### Step 3 - Costs and learning

Once you've fed in all of the information for a set of inputs, you can check out the test_set_costs which is a list of the costs for each of the training examples that you've used.

Next, you'll want to get your network to improve on what it has done! This is where the learn() function comes in. The learn function will adjust all of your weights and biases based with the weights_costs and bias_costs attributes that we've calculated using backpropagation, and then reset those attributes ready for the next training set.

The cost_per_training_set will record the average cost of each training set, so that you can see how your network is improving with each training set. 

## Improvement

 - Network GUI to give a visual representation of the network and it's progress.

