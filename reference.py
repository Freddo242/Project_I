import numpy as np
from matplotlib import pyplot as plt
import pygame
import math

def collide():

        pass

def vect_len(pointa,pointb):
    return math.sqrt( (pointa[0]-pointb[0])**2 + (pointa[1]-pointb[1])**2 )

class particle:

    def __init__(self, mass, radius, velocity, sides, position=(0,0)):
        self.mass = mass
        self.r = radius
        self.velocity = velocity
        self.n = sides
        self.vertices = []
        self.iscircle = False
        self.pos = position

        #defining vertices.
        if self.n < 3:
            iscircle = true
        else:
            angle = (2*math.pi)/self.n
            for i in range(0,self.n):
                self.vertices.append( ( round(self.pos[0]+self.r*math.cos(i*angle),5) , round(self.pos[1]+self.r*math.sin(i*angle),5) ) )

    def Move():
        pass


def main():
    square = particle(1,1,1,4)
    decagon = particle(1,1,1,10)
    print(square.vertices)
    print(decagon.vertices)
    for i in range(0,decagon.n):
        print(vect_len(decagon.vertices[i-1],decagon.vertices[i]))

if __name__ == "__main__":
    main()
