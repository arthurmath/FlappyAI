import matplotlib.pyplot as plt
from game import Session
from pathlib import Path
import copy as cp
import numpy as np
import random as rd
import pickle
import os

    
POPULATION = 500
SURVIVAL_RATE = 0.1
N_GENERATIONS = 20
SEED = 42
MUTATION_RATE = 0.1
STD_MUTATION = 0.3
NN_LAYERS = [4, 5, 5, 1]

np.random.seed(SEED)
rd.seed(SEED)



class Pilot():
   
    def __init__(self, weights=None, biases=None):
    
        if weights != None :
            self.weights = cp.deepcopy(weights)
        else:
            self.initialize_weights()
            
        if biases != None:
            self.bias = cp.deepcopy(biases)
        else:
            self.initialize_bias()
        
            
    def initialize_weights(self):
        self.weights = []
        for i in range(len(NN_LAYERS) - 1): 
            # Pour chaque couche du NN, creation d'une matrice de poids 
            layer = [[rd.uniform(-1, 1) for _ in range(NN_LAYERS[i+1])] for _ in range(NN_LAYERS[i])] # rd.gauss(0, 0.5)
            self.weights.append(np.matrix(layer))
            
        
    def initialize_bias(self):
        self.bias = []
        for layer in self.weights:
            nbrBias = np.size(layer, axis=1)
            self.bias.append(np.array([rd.uniform(-1, 1) for _ in range(nbrBias)])) # rd.gauss(0, 0.5)
            
    
    def predict(self, vector):
        for weight, bias in zip(self.weights, self.bias):
            vector = np.dot(np.array(vector), np.matrix(weight)) + np.array(bias)
            vector = self.heaviside(vector)

        return vector
    
    def heaviside(self, x):
        return (x > 0).astype(int)
    

    # Pour l'entrainement
    
    def mate(self, other):
        """ Mix the copy of this DNA with the copy of another one to create a new one. """
        newWeights = self.crossover(self.weights, other.weights)
        newBias = self.crossover(self.bias, other.bias)
        return Pilot(newWeights, newBias)
    
    
    def crossover(self, dna1, dna2):
        """ Performs a crosover on the layers (weights and biases) """
        res = [self.cross_layer(dna1[layer], dna2[layer]) for layer in range(len(dna1))]
        return res

    def cross_layer(self, layer1, layer2): # better
        """ Performs a crossover on two layers """
        lineCut = rd.randint(0, layer1.shape[0] - 1)
        if len(layer1.shape) == 1:  # 1D case
            return np.hstack((layer1[:lineCut], layer2[lineCut:]))

        columnCut = rd.randint(0, layer1.shape[1] - 1)
        res = np.vstack((
            layer1[:lineCut],
            np.hstack((layer1[lineCut, :columnCut], layer2[lineCut, columnCut:])),
            layer2[lineCut + 1 :],
            ))
        return res
    

    




class GeneticAlgo:
    
    def train(self):
        
        self.list_scores = []
        
        self.population = [Pilot() for _ in range(POPULATION)]

        for self.generation in range(N_GENERATIONS):
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            if self.ses.quit:
                break
            
            print(f"Generation {self.generation+1}, average score: {self.avgGenScore:.0f}, best score: {self.bestGenScore}")

        if not self.ses.quit:
            self.evaluate_generation() # Evaluate the last generation
            self.bests_survives()
            self.bestPilotEver = self.bestPilots[0]
        


    def evaluate_generation(self):
            
        self.ses = Session(POPULATION, self.generation)
        states = self.ses.reset()

        while not self.ses.done:
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
            actions = [mat.tolist()[0][0] for mat in actions]
            states, self.scores, _ = self.ses.step(actions)
            
        self.bestGenScore = max(self.scores)
        self.avgGenScore = sum(self.scores) / POPULATION
        self.list_scores.append(self.bestGenScore)
        


            
    def bests_survives(self):
        
        sorted_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)
        
        population_sorted = [self.population[i] for i in sorted_indices] 
        scores_sorted = [self.scores[i] for i in sorted_indices] 
        
        self.bestPilots = population_sorted[:int(POPULATION * SURVIVAL_RATE)] # take the 10% bests pilots
        self.bestscores = scores_sorted[:int(POPULATION * SURVIVAL_RATE)]  # take the 10% bests scores
                
                
                
    def change_generation(self):
        """ Creates a new generation of pilot. """
        
        self.new_population = cp.copy(self.bestPilots) # 10% best pilots
        
        while len(self.new_population) < POPULATION:
            parent1, parent2 = self.select_parents()
            baby = parent1.mate(parent2)
            self.new_population.append(baby)
        
        self.population = self.new_population
        
    
    def select_parents(self):
        """Select two pilots with high scores."""
        total_scores = sum(self.bestscores)
        ratios = [f / total_scores for f in self.bestscores]
        return rd.choices(self.bestPilots, weights=ratios, k=2) # return a k-sized list # weights=ratios,











if __name__ == "__main__":
    
    
    algo = GeneticAlgo()
    algo.train()
    
    print("\nBests scores total: ", sum(algo.list_scores), "\n")
    
    
    
    # # Save the weights and biases of the snakes for the new game scores
    # files = os.listdir(Path("weights"))
    # n_train = len(files) # nb de fichiers dans dossier weights
    # with open(Path("weights") / Path(f"{n_train}.weights"), "wb") as f: # write binary
    #     pickle.dump((algo.bestPilotEver.weights, algo.bestPilotEver.bias), f)
        
        
    
    # # Show graph of progressions
    # plt.plot(algo.list_scores)
    # plt.xlabel("Générations")
    # plt.ylabel("Progression (%)")
    # plt.show()
    
    
    
    
    
# Base : (self.bestPilots, weights=ratios), NN_LAYERS = [4, 3, 3, 1], Mutate useless
# Generation 1, average score: 24, best score: 96
# Generation 2, average score: 29, best score: 106
# Generation 3, average score: 45, best score: 265
# Generation 4, average score: 40, best score: 257
# Generation 5, average score: 39, best score: 142
# Generation 6, average score: 29, best score: 177
# Generation 7, average score: 64, best score: 1387
# Generation 8, average score: 189, best score: 2266

# self.bestPilots
# Generation 1, average score: 24, best score: 96
# Generation 2, average score: 28, best score: 117
# Generation 3, average score: 41, best score: 1347
# Generation 4, average score: 54, best score: 2547

# NN_LAYERS = [4, 5, 5, 1]
# Generation 1, average score: 26, best score: 177
# Generation 2, average score: 33, best score: 904
# Generation 3, average score: 38, best score: 217
# Generation 4, average score: 79, best score: 907
# Generation 5, average score: 308, best score: 625
# Generation 6, average score: 158, best score: 1027
# Generation 7, average score: 271, best score: 944
# Generation 8, average score: 151, best score: 906
# Generation 9, average score: 125, best score: 146

# np.array (pilot line 43)
# Generation 1, average score: 24, best score: 96
# Generation 2, average score: 29, best score: 179
# Generation 3, average score: 44, best score: 266
# Generation 4, average score: 45, best score: 182
# Generation 5, average score: 64, best score: 385
# Generation 6, average score: 62, best score: 340
# Generation 7, average score: 64, best score: 666
# Generation 8, average score: 109, best score: 1499