import numpy as np
from mathssss import multiply_two_dim , vecxmatrix , add_vector, dot_prod , sigmoid , d_sigmoid
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

        self.bprop_matrices = [0 for i in range(self.L)]
        self.c_bprop_matrices = [0 for i in range(self.L)]

        self.sig_prime_z = [0 for i in range(self.L)]

        self.test_set_costs = []
        self.c_by_last_layer = []
        self.cost_per_training_set = []

        self.weights_cost = [ np.array([ [ 0 for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ]).astype('float32') for layer in range(self.L - 1) ]
        self.weights_cost.insert(0,0)

        self.bias_cost = [ np.array( [ 0 for i in range(self.layers[j])] ).astype('float32') for j in range(self.L) ]
        

    def gen_B(self):

        #Generates a list of arrays the same length as the layers in the network, initialising with random numbers between -1 and 1

        return [ np.array( [ random.uniform(-1,1) for i in range(self.layers[j])] ) for j in range(self.L) ]


    def gen_weights(self):

        #generates a list of transformation matrices initialised with random numbers between -1 and 1

        return [ np.array([ [ random.uniform(-1,1) for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ]) for layer in range(self.L - 1) ]
        

    def forward_propagation(self, given_input , y):

        #checks the inputs and raises ValueErrors if they're not the correct length
        
        if len(given_input) != self.layers[0]:
            raise ValueError("Expected vector does not equal length of input layer")
        else: 
            self.A[0] = np.array(given_input)


        if len(y) != self.layers[-1]:
            raise ValueError("Expected vector does not equal length of output layer")
        else: 
            self.y = np.array(y)

        #Calculates the value of each node from the given_input. Generates Z.

        for layer in range(1,self.L):

            Z = add_vector(vecxmatrix(self.weights[layer],self.A[layer-1]) , self.B[layer] )
            self.Z[layer] = Z
            self.A[layer] = np.array([sigmoid(x) for x in Z]).astype('float32')


        #calculating the cost of network
        new_cost = 0
        for i in range(self.layers[-1]):
            new_cost += (self.A[-1][i] - self.y[i])**2
        self.test_set_costs.append(new_cost)


        #calculations for backpropagation
        self.c_by_last_layer = np.array( [ (2*(self.A[-1][i] - self.y[i])) for i in range(self.layers[-1]) ] )

        self.gen_backprop_matrices()
        self.gen_cbprop_matrices()


    def calc_output(self,input):

        if len(input) != self.layers[0]:
            raise ValueError("Expected vector does not equal length of input layer")
        else: 
            self.A[0] = np.array(input)

        for layer in range(1,self.L):

            Z = add_vector(vecxmatrix(self.weights[layer],self.A[layer-1]) , self.B[layer] )
            self.Z[layer] = Z
            self.A[layer] = np.array([sigmoid(x) for x in Z]).astype('float32')

        return self.A[-1]


    def gen_sig_prime_z(self):
        
        for layer in range(1,self.L):

            vec = np.array( [ d_sigmoid(self.Z[layer][i]) for i in range(len(self.Z[layer])) ] ).astype('float32')

            self.sig_prime_z[layer] = vec



    def gen_backprop_matrices(self):

        self.gen_sig_prime_z()

        for layer in range(2,self.L):
            matrix = []

            for i in range(len(self.A[layer])):  
                vec = []

                for j in range( len(self.A[layer-1])):


                    vec.append( self.sig_prime_z[layer][i] * self.weights[layer][i][j] )

                matrix.append(vec)

            self.bprop_matrices[layer] = np.array(matrix).astype('float32')



    def gen_cbprop_matrices(self):

        matrix = self.bprop_matrices[-1]
        self.c_bprop_matrices[-1] = matrix

        for i in range(2,self.L - 1):
            matrix = multiply_two_dim(matrix, self.bprop_matrices[-i])
            self.c_bprop_matrices[-i] = matrix
        


    def gen_cost_by_weight(self, layer, from_index, to_index):

        if layer == self.L - 1:

            return self.c_by_last_layer[to_index] * self.sig_prime_z[-1][to_index] * self.weights[-1][to_index][from_index]

        if layer == self.L - 2:

            term_a = self.sig_prime_z[-2][to_index] * self.A[-3][from_index]

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



    def gen_cost_gradient(self):

        for layer in range(1,self.L):
            
            for i in range(len(self.A[layer])):

                bias_adjustment = self.gen_cost_by_bias(layer,i)
                #This is minus because we want the negative gradient of the cost function. Same goes for weights adjustment.
                self.bias_cost[layer][i] -= bias_adjustment

                for j in range(len(self.A[layer-1])):

                    adjustment = self.gen_cost_by_weight(layer,j,i)

                    self.weights_cost[layer][i][j] -= adjustment

        


    def learn(self):

        #Here we need to divide each cost value by the number of training examples.
        for matrix in range(1,len(self.weights)):
            for vector in range(len(self.weights_cost[matrix])):
                for i in range(len(self.weights_cost[matrix][vector])):
                    self.weights_cost[matrix][vector][i] = (1/len(self.test_set_costs))*self.weights_cost[matrix][vector][i]

        #And the same for the biases
        for vector in range(1,len(self.bias_cost)):
            for element in range(len(self.bias_cost[vector])):
                self.bias_cost[vector][element] = (1/len(self.test_set_costs))*self.bias_cost[vector][element]


        #adding the gradient function to the weights and biases
        for layer in range(1,self.L):
            for i in range(len(self.A[layer])):
                self.B[layer][i] += self.bias_cost[layer][i]
                for j in range(len(self.A[layer-1])):
                    self.weights[layer][i][j] += self.weights_cost[layer][i][j]

        average_cost = 0
        for cost in range(len(self.test_set_costs)):
            average_cost += self.test_set_costs[cost]
        self.cost_per_training_set.append(average_cost/len(self.test_set_costs))

        self.reset_costs()


    def reset_costs(self):

        #reseting the weights and biases costs
        self.weights_cost = [ np.array([ [ 0.0 for j in range(self.layers[layer]) ] for i in range(self.layers[layer+1]) ]).astype('float32') for layer in range(self.L - 1) ]
        self.weights_cost.insert(0,0)
        self.bias_cost = [ np.array( [ 0.0 for i in range(self.layers[j])] ).astype('float32') for j in range(self.L) ]

        self.test_set_costs = []


    def rate_of_convergence(self):
        
        pass

    def save(self,file_path):
        #Save weights and biases to a file
        pass

    def load(self,file_path):
        #loads weights and biases from a file
        pass



def main():
    
    network = NeuralNetwork([4,7,7,7,4])
    


if __name__ == "__main__":
    main()
