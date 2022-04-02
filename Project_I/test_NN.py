import NeuralNetworks_2 as NN 



def main():

    small_network = NN.NeuralNetwork([2,3,2])
    medium_network = NN.NeuralNetwork([3,4,4,3])
    large_network = NN.NeuralNetwork([3,4,5,4,3])

    unit_tests()

    return 0

def unit_tests(small_network,medium_network,large_network):

    small_network.forward_propogation([0,0] , [1,0])
    #checks
    small_network.forward_propogation([1,1] , [0,1]) 
    #checks
    medium_network.forward_propogation([0,0,0] , [1,0,0])
    #checks
    medium_network.forward_propogation([1,1,0] , [0,0,1])
    #checks
    large_network.forward_propogation([1,0,1] , [0,1,0])
    #checks
    large_network.forward_propogation([1,1,1] , [0,1,0])
    #checks


if __name__ == "__main__":

    main()

