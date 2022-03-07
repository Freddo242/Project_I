from doctest import OutputChecker
from locale import nl_langinfo
import math
import numpy as np
from mathssss import multiply_two_dim , vecxmatrix , add_vector, dot_prod , sigmoid , d_sigmoid , transpose

#Some notation to make this simpler. There are L layers (including input and output), the weights ( W(L) ) for layer L are the weights that take a(L-1) to a(L)

class NeuralNetwork:

    def __init__(self, input, output, hidden):
        
        #All of the arguments should be single integers. After instansiating the class, they should run get_input() to begin the network.
        if type(input) != int or type(output) != int:
            raise ValueError("Input and Output arguments should be integers")

        if type(hidden) != list:
            raise ValueError("hidden values should be a vector of the number of nodes at each level")
        

        self.input_vec = [0 for i in range(input)] 
        self.output_vec = [0 for i in range(output)]

        '''I have seperate hidden vectors for hidden nodes rather than a big list of nodes and their values because the input and outputs are special layers that I was to be able to easily differentiate between and apply functions to. 
        This may cause problems down the line since I will need to remember to apply funcitons independantly to inputs and outputs.
        I can change this another time.
        '''
        self.hidden_vec = [ [1 for j in range(hidden[i]) ] for i in range(len(hidden)) ]

        #There will be L-1 matricies in this weights vector.
        self.weights = []

        #The backprop matricies are going to help us learn!
        self.backprop_matricies = []
        self.cum_backdrop_matricies = []

        #for z_L there is going to be 1 less layer since we don't have a z_L for the first layer. Z = W*a - b
        self.z_L = [ [0 for i in range(hidden[j])] for j in range(len(hidden)) ] + [ [0 for i in range(output)] ]

        self.bias = [[] for i in range(len(hidden) + 1)]


        #Creating the weights matrix. This is a 3 dimensional matrix: a list of lists of transformation vectors. There may be a better way to do this.
        self.weights.append([[1 for i in range(len(self.input_vec))] for j in range(hidden[0])])
        for i in range(len(hidden)-1):
           self.weights.append([ [1 for i in range(hidden[i])] for j in range(hidden[i+1])])

        self.weights.append([[1 for i in range(len(self.weights[-1]))] for j in range(len(self.output_vec))]) 

        #Creating the bias'. This is a vector of other different sized vectors.
        for i in range(len(self.bias)-1):
            self.bias[i] = [0 for j in range(hidden[i])]
        self.bias[-1] = [0 for j in range(output)]

        self.error_vec = []

    def get_W_L(self,layer):
        #returns the weights matrix for a given layer

        try:
            return self.weights[layer]
        except IndexError:
            print("The layer is out of range. The first layer starts from 0.")
    
    def get_b_L(self,layer):
        #returns the bias vector for a given layer

        try:
            print(self.bias[layer])
        except IndexError:
            print("The layer is out of range. The first layer starts from 0.")


    def print_a_L(self, layer):

        if layer == 0:
            print("layer ", str(layer), self.input_vec)
        elif layer == len(self.hidden_vec) - 1:
            print("layer ", str(layer), self.output_vec)
        else:
            print("layer ",str(layer), self.hidden_vec[layer-1])


    def gen_output(self):

        '''There are two ways that I could hav generated an output. 
        One would be using the method below -- dot producting vectors -- and the other is using it as linear transformations. 
        i.e. applying the linear transformation W to each A[i]'''


        input_and_hidden_nodes = [self.input_vec] + self.hidden_vec


        for i in range(len(self.hidden_vec)):

            for j in range(len(self.hidden_vec[i])):

                #Each node is the dot product of the previous layer and the corresponding weights.
                z = dot_prod(input_and_hidden_nodes[i],self.weights[i][j]) - self.bias[i][j]
                self.z_L[i][j] = z
                self.hidden_vec[i][j] = sigmoid(z)


        for i in range(len(self.output_vec)):

            z = dot_prod(input_and_hidden_nodes[-1],self.weights[-1][i]) - self.bias[-1][i]
            self.z_L[-1][i] = z
            self.output_vec[i] = sigmoid(z)


    def get_input(self, input):

        if type(input) != list:
            raise ValueError("You did not give a vector")
        if len(input) != len(self.input_vec):
            raise ValueError("This vector does not have the correct number of nodes for the input of this network.")
        self.input_vec = input


    def calc_cost(self,expected_output):

        '''This is going to be the average of the sum of the squares of the error'''

        cost = 0

        if len(expected_output) != len(self.output_vec):
            raise ValueError("Expected output length does not match initialised output length")
        self.gen_output()

        output = self.output_vec     #we don't want to make any changes to self.output_vec,so will use output instead.

        for element in output:
            element = -element
        #because of the last line, this next line is essentially the same as v - w a]

        self.error_vec = add_vector(expected_output , output)

        for i in self.error_vec:
            cost += i**2

        cost = cost / len(self.error_vec)

        return cost


    # def gen_d_cost(self):

    #     COMMENTING THIS OUT FOR THE TIME BEING. MAY NOT BE NEEDED

    #     for i in self.error_vec:
    #         self.d_cost_by_final_layer += 2*i

    #     self.d_cost_by_final_layer = self.d_cost_by_final_layer / len(self.output_vec)

    def d_layer_by_prev_layer(self,layer):

        '''This function is to generate "The term" (diff sigmoid of Zi transformed by matrix weight for that layer)
        This is the d_a_L / d_a_L-1 term. i.e. the differential of a on layer L by the differential of A on layer L-1.
        The term is a vector of length len(layer).  
        '''

        n_l = len(self.z_L[layer])
        n_l_1 = len(self.z_L[layer - 1])

        #print("n_1, n_l_1 ", n_l , n_l_1) output was 10 , 8

        diff_sig_z = []
        for i in range(len(self.z_L[layer])):
            diff_sig_z.append(d_sigmoid(self.z_L[layer][i]))

        weight = self.weights[layer]

        W = [ [0 for i in range(n_l_1)] for j in range(n_l) ]

        for p in range(n_l):

            for k in range(n_l_1):

                W[p][k] = diff_sig_z[p]*weight[p][k]

        return W
    

    def gen_backprop_matricies(self):

        '''Get all of the backprop matricies.
        This is going to be one for every layer from L-1 to L-n+2 (don't need to do the last two layers. one is output, the other won't get used in this way). 
        Once we have them all, we can use them in our calculations to find the gradient function'''

        for layer in range(len(self.hidden_vec)):

            self.backprop_matricies.append(self.d_layer_by_prev_layer(layer+1))



    def gen_cum_backprop(self):

        '''This is going to make calculations a little easier. We only want to do matrix multiplication once really to save time.
        So, for this function, I want to generate the cumulative backprop matrix, so we can pick it up at any layer.'''

        pass


def main():

    #Create the network
    network = NeuralNetwork(7,10,[8,8])
    #network.print_hidden()
    network.get_input([1,1,0,1,0,1,1])
    network.gen_output()
    #print(" Output: " , network.output_vec)
    #print("Hidden: ", network.hidden_vec)

    #print("calculate cost function: " , network.calc_cost([1,0,1,0,1,0,0,1,0,1]))

    network.gen_backprop_matricies()
    for i in range(2):
        print("matrix " , i)
        print("len(matrix) , len(matrix[0]) " , len(network.backprop_matricies[-1-i]) , len(network.backprop_matricies[-1-i][0]))
     
     




if __name__ == "__main__":
    main()
