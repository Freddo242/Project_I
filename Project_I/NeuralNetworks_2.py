import math
from typing import final
import numpy as np
from mathssss import multiply_two_dim , vecxmatrix , add_vector, dot_prod , sigmoid , d_sigmoid , transpose
import random


class NeuralNetwork:

    def __init__(self,layers):

        self.layers = layers
        self.L = len(layers)
        self.A = [ 0 for i in range(self.L) ]
        self.weights = self.gen_weights()
        self.weights.insert(0,0)
        self.Z = [ 0 for i in range(self.L) ]
        self.B = self.gen_B()

        self.y = 0

        self.bprop_matrices = [0,0]
        self.c_bprop_matrices = [0,0]

        self.sig_prime_z = [0]

        self.c_by_last_layer = []
        

    def gen_B(self):

        return [ np.array( [ random.uniform(-1,1) for i in range(self.layers[j])] ) for j in range(self.L) ]


    def gen_weights(self):

        return [ np.array([ [ random.uniform(-1,1) for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ]) for layer in range(self.L - 1) ]
        

    def forward_propogation(self, given_input ,y ):

        
        if len(given_input) != self.layers[0]:
            raise ValueError("Expected vector does not equal length of input layer")
        else: 
            self.A[0] = np.array(given_input)


        if len(y) != self.layers[-1]:
            raise ValueError("Expected vector does not equal length of output layer")
        else: 
            self.y = np.array(y)


        for layer in range(1,self.L):

            Z = add_vector(vecxmatrix(self.weights[layer],self.A[layer-1]) , self.B[layer] )
            self.Z[layer] = Z
            self.A[layer] = np.array([sigmoid(x) for x in Z])
            
        self.c_by_last_layer = np.array( [ (2*(self.A[-1][i] - self.y[i]))/(self.layers[-1]) for i in range(self.layers[-1]) ] )


    def gen_sig_prime_z(self):
        
        for layer in range(1,self.L):

            vec = np.array( [ d_sigmoid(self.Z[layer][i]) for i in range(len(self.Z[layer])) ] ).astype('float64')

            self.sig_prime_z.append(vec)


    def gen_backprop_matrices(self):

        self.gen_sig_prime_z()

        for layer in range(2,self.L):
            matrix = []

            for i in range(len(self.A[layer])):  
                vec = []

                for j in range( len(self.A[layer-1])):

                    #print("layer ,i, j", layer, i , j)
                    vec.append( self.sig_prime_z[layer][i] * self.weights[layer][i][j] )

                matrix.append(vec)

            self.bprop_matrices.append(np.array(matrix).astype('float64'))


    def gen_cbprop_matrices(self):

        matrix = self.bprop_matrices[-1]
        self.c_bprop_matrices.append(matrix)

        for i in range(2,self.L - 1):
            matrix = multiply_two_dim(matrix, self.bprop_matrices[-i])
            self.c_bprop_matrices.insert(2, matrix)
        

    def gen_cost_by_weight(self, layer, from_index, to_index):

        if layer == self.L - 1:

            return self.c_by_last_layer[to_index] * self.sig_prime_z[-1][to_index] * self.weights[-1][to_index][from_index]

        if layer == self.L - 2:

            term_a = self.sig_prime_z[-2][to_index] * self.A[-2][from_index]

            term_w = np.array([ self.sig_prime_z[-1][i] * self.weights[-1][i][to_index] for i in range(self.layers[-1]) ])

            result_b = dot_prod(self.c_by_last_layer , term_w)

            result_b *= term_a

            return result_b

        else:

            vec = np.array([ self.sig_prime_z[layer+1][i] * self.weights[layer+1][i][to_index] for i in range(len(self.Z[layer+1])) ])

            scalar = self.sig_prime_z[layer][to_index]*self.A[layer-1][from_index]

            matrix = self.c_bprop_matrices[-(self.L-2-layer)]

            matrix_on_vec = vecxmatrix( matrix , vec )

            result_c = dot_prod(matrix_on_vec , self.c_by_last_layer)

            result_c *= scalar

            return result_c


    def gen_cost_by_bias(self, layer, index):

        if layer == self.L - 1:

            return self.c_by_last_layer[index] * self.sig_prime_z[-1][index]

        if layer == self.L - 2:

            scalar = self.sig_prime_z[-2][index]

            mid_vec = np.array([ self.sig_prime_z[-1][i] * self.weights[-1][i][index] for i in range(self.layers[-1]) ])

            result_b = dot_prod(mid_vec, self.c_by_last_layer)

            result_b *= scalar

            return result_b

        else:

            vec = np.array([ self.sig_prime_z[layer+1][i] * self.weights[layer+1][i][index] for i in range(len(self.Z[layer+1])) ])

            scalar = self.sig_prime_z[layer][index]

            matrix = self.c_bprop_matrices[-(self.L-2-layer)]

            matrix_on_vec = vecxmatrix( matrix , vec )

            result_c = dot_prod(matrix_on_vec , self.c_by_last_layer)

            result_c *= scalar

            return result_c



    def adjust_weights_and_bias(self):

        for layer in range(1,self.L):

            for i in range(len(self.A[layer])):

                bias_adjustment = self.gen_cost_by_bias(layer,i)

                self.B[layer][i] -= bias_adjustment

                for j in range(len(self.A[layer-1])):

                    adjustment = self.gen_cost_by_weight(layer,j,i)

                    self.weights[layer][i][j] -= adjustment

        
    def learn(self):
        self.gen_backprop_matrices()
        self.gen_cbprop_matrices()


        self.adjust_weights_and_bias()
        print(self.A[-1])



def main():
    
    network = NeuralNetwork([2,3,2,3,2])

    #network.forward_propogation( [1,0], [0,1])

    #network.gen_backprop_matrices()
    #network.gen_cbprop_matrices()

    #print(network.gen_cost_by_weight(4,2,0))
    #print(network.gen_cost_by_weight(3,1,0))
    #print(network.gen_cost_by_weight(2,1,1))
    #print(network.gen_cost_by_bias(4,0))
    #print(network.gen_cost_by_bias(3,0))
    #print(network.gen_cost_by_bias(2,1))

    for i in range(100):
        network.forward_propogation( [1,0],[0,1] )
        network.learn()

    


if __name__ == "__main__":
    main()
