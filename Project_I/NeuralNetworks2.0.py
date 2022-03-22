import math
from typing import final
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

        self.cost_by_weights = self.gen_weights()
        self.cost_by_weights.insert(0,0)
        self.cost_by_bias = [np.array([ 0 for i in range(layer) ]).astype('float64') for layer in layers]

        self.bprop_matrices = [0,0]
        self.c_bprop_matrices = [0,0]

        self.sig_prime_z = []

        self.c_by_last_layer = []
        

    def gen_A(self):

        return [ np.array([self.beginning_node_value for i in range(self.layers[j])] ).astype('float64') for j in range(self.L) ]


    def gen_weights(self):

        return [ np.array([ [ 1 for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ]).astype('float64') for layer in range(self.L - 1) ]


    def forward_propogation(self, y ):
        
        if len(y) != self.layers[-1]:
            raise ValueError("Expected vector does not equal length of output layer")
        else: 
            self.y = np.array(y).astype('float64')

        for layer in range(1,self.L):

            Z = add_vector(vecxmatrix(self.weights[layer],self.A[layer-1]) , np.array([ -element for element in self.B[layer] ]) )
            self.Z[layer] = Z
            self.A[layer] = np.array([sigmoid(x) for x in Z])
            
        self.c_by_last_layer = np.array( [ (2*(self.A[-1][i] - self.y[i]))/(self.layers[-1]) for i in range(self.layers[-1]) ] )


    def gen_sig_prime_z(self):
        
        for layer in range(self.L):

            vec = np.array( [ d_sigmoid(self.Z[layer][i]) for i in range(len(self.Z[layer])) ] ).astype('float64')

            self.sig_prime_z.append(vec)


    def gen_backprop_matrices(self):

        self.gen_sig_prime_z()

        for layer in range(2,self.L):
            matrix = []

            for i in range(self.layers[layer-1]):
                vec = []

                for j in range(self.layers[layer]):
                    vec.append( self.sig_prime_z[layer][j] * self.weights[layer][j][i] )

                matrix.append(vec)

            self.bprop_matrices.append(np.array(matrix).astype('float64'))


    def gen_cbprop_matrices(self):

        matrix = self.bprop_matrices[-1]
        self.c_bprop_matrices.append(matrix)

        for i in range(2,self.L - 1):
            matrix = multiply_two_dim(self.bprop_matrices[-i],matrix)
            self.c_bprop_matrices.insert(2, matrix)
        

    def gen_cost_by_weight(self, layer, from_index, to_index):

        if layer == self.L - 1:

            return self.c_by_last_layer[to_index] * self.sig_prime_z[-1][to_index] * self.weights[-1][to_index][from_index]

        if layer == self.L - 2:

            term_a = self.sig_prime_z[-2][to_index] * self.A[-2][from_index]

            term_w = [ self.sig_prime_z[-1][i] * self.weights[-1][i][to_index] for i in range(self.layers[-1]) ]

            result_b = dot_prod(self.c_by_last_layer , term_w)

            result_b *= term_a

            return result_b

        else:

            final_term = np.array([ self.sig_prime_z[layer+1][j] * self.weights[layer+1][j][from_index] for j in range(self.layers[layer+1]) ] ).astype('float64')

            transform_matrix_on_final_term = vecxmatrix(transpose(self.c_bprop_matrices[-layer]) , final_term)

            result_c = dot_prod(transform_matrix_on_final_term,self.c_by_last_layer) * self.sig_prime_z[layer][to_index] * self.A[layer-1][from_index]        

            return result_c


    def adjust_nodes(self):

        for layer in range(1,self.L):

            for i in range(len(self.weights[layer])):

                for j in range(len(self.weights[layer][i])):

                    print("Adjust: " , layer, i , j)
                    adjustment = self.gen_cost_by_weight(layer,j,i)
                    self.weights[layer][i][j] += adjustment
        


def main():
    
    network = NeuralNetwork([4,5,6,3])
    network.forward_propogation([1,0,0])
    network.gen_backprop_matrices()
    network.gen_cbprop_matrices()
    print("weights before" , network.weights)
    network.adjust_nodes()
    print("weights after" , network.weights)
  

if __name__ == "__main__":
    main()
