# Project_I: Neural Networks

A Neural Network class that can be used to initialise a network of any size, calculate outputs, and learn via backpropogation. 

I started this project after watching 3Blue1Brown's serires on the Neural Networks (https://www.3blue1brown.com/topics/neural-networks). The maths is really interesting: I loved Linear Algebra in Uni! I was initially intending to do a project on something else entirely, but I got caught up in trying to get a formula for the gradient of the cost function for any size of network. Naturally, when I had done this, I had to see if I could put it into Python: and it worked!

It still needs some tweaking to make it usable, but all of the important components needed to make a network learn are there.

## Table of contents
*[General info](#General-info)
*[Technologies](#Technologies)
*[Setup](*Setup)

## General info

This is primarily a learning project for me; I went into this project completely ignorant of how to code a Neural Network and I wanted to see how far I could get from just knowing the basic principles behind back-propagation using the formulas that I had derived. There are definitely more sophisticated ways to do it, but I hope anyone using this can gain some insight from a complete newbie's interpretation of Neural Networks.

The idea is that you can create a neural network and fairly easily keep track of what it is doing. If you're new to Neural Networks, like I am, then you  may want to know what the weights and bias' are doing at particular points ; you may want to see what the cost of a particular weight would be with respect to the cost function; and I hope that it is easy to do so with the attributes and methods that the class provides.

## Technologies
Project created with:
 - Python 3.10.0
 - numpy 1.21.4
 
## Setup

 - have the Mathssss.py in the same directory.

## Usage

 - 

## Improvement

 - Fix bug which appears when you create a network with a larger than two node difference between any middle layers.
 - Add function that will read a csv and learn from the set of data
 - Add function that will read a csv and give an output without adjusting the weights and bias'
 - Make more user friendly.
 - Network GUI to give a visual representation of the network and it's progress.

