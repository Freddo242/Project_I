import math
import numpy as np
from mathssss import multiply_two_dim , vecxmatrix , add_vector, dot_prod , sigmoid , d_sigmoid , transpose


class NeuralNetwork:

    def __init__(self,layers):

        self.layers = layers
        self.beginning_node_value = 0.5
        self.L = len(layers)

        self.A = self.gen_A()
        self.weights = self.gen_weights()
        self.weights.insert(0,0)
        self.Z = self.gen_A()
        self.B = self.gen_A()

        self.y = 0
        cost_by_weights = np.array([])
        cost_by_bias = np.array([])

        bprop_matrices = np.array([ [] for i in range(self.L) ])
        c_bprop_matrices = np.array([ [] for i in range(self.L) ])
        

    def gen_A(self):

        return np.array([ [self.beginning_node_value for i in range(self.layers[j])] for j in range(self.L) ])


    def gen_weights(self):

        return np.array([ [ [ 1 for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ] for layer in range(self.L - 1) ])


    def forward_propogation(self, y ):
        
        if len(y) != self.layers[-1]:
            raise ValueError("Expected vector does not equal length of output layer")
        else: 
            self.y = y


        for layer in range(1,self.L):

            Z = add_vector(vecxmatrix(self.weights[layer],self.A[layer-1]) , [ -element for element in self.B[layer] ] )
            self.Z[layer] = Z
            self.A[layer] = [sigmoid(x) for x in Z]
            



def main():
    
    network = NeuralNetwork([4,5,6,3])
    network.forward_propogation()
    


if __name__ == "__main__":
    main()
