import random as rd
import pickle
import os
import matplotlib.pyplot as plt
from old_pilot import Pilot
from old_game import Session
from pathlib import Path
import copy as cp


    
POPULATION = 500
SURVIVAL_RATE = 0.1
N_GENERATIONS = 20

# Autres paramètres :
# nombre de layers NN
# fonctions activation 




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
            baby.mutate() # useless
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