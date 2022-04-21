# Project_I: Neural Networks

A Neural Network class that can be used to initialise a network of any size, calculate outputs, and minimise the cost function. 

I started this project after watching 3Blue1Brown's serires on the Neural Networks (https://www.3blue1brown.com/topics/neural-networks). The maths is really interesting: I loved Linear Algebra in Uni! I was initially intending to do a project on something else entirely, but the maths was fun and once I had done that, I had to see if I could put it into Python: and it worked!

It still needs some tweaking to make it usable, but all of the important components needed to make a network learn are there.

Feel free to correct me where I have made a mistake! The only way I know this is correct is because it somewhat works.

## Table of contents
*[General info](#General-info)
*[Technologies](#Technologies)
*[Setup](*Setup)

## General info

This is primarily a learning project for me; I went into this project completely ignorant of what a Neural Network is and I wanted to see how far I could get from just knowing the basic principles behind back-propagation and using formulas I had derived. There are definitely more sophisticated ways to do it, but I hope anyone using this can gain some insight from a complete novice's interpretation of Neural Networks. 

## Technologies
Project created with:
 - Python 3.10.0
 - numpy 1.21.4
 
## Setup
There are 3 files in this project. NeuralNetworks.py was my first attempt, before I had standardised any of the notation, and I've left it there in case anyone is interested to see how I began.

NeuralNetworks_2.py contains the NeuralNetworks class that you'll want to use. This needs to sit inside a directory with mathssss.py, since it imports functions from that script.

Once you have downloaded them, you can just use the import keyword in Python to use the class like you would any other module.

## Usage

### Step 1 - Initialise a NeuralNetwork
Network = NeuralNetwork( [layers] )
the init function takes in one argument, a list of the number of nodes in each layer. 

e.g. NeuralNetwork( [4,5,6,5,4] ) is a network with 5 layers with the nodes on each layer corresponding to the values of the list respectively.

### Step 2 - Feeding in data

This is TBC for now since it needs some refinement before we can throw data at it!

## Improvement

 - Fix bug which appears when you create a network with a larger than two node difference between any middle layers.
 - Create a function to store the cost function adjustment rather than adjust the network after every calculation of the gradient function. 
 - Add function that will read a csv and learn from the set of data
 - Add function that will read a csv and give an output without adjusting the weights and bias'
 - Make more user friendly.
 - Network GUI to give a visual representation of the network and it's progress.

