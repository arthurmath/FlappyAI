import random as rd
import pickle
import os
import matplotlib.pyplot as plt
from pilot import Pilot
from game import Session
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
            
        self.ses = Session(POPULATION, self.generation, display=False)
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
            baby.mutate()
            self.new_population.append(baby)
        
        self.population = self.new_population
        
    
    def select_parents(self):
        """Select two pilots with high scores."""
        total_scores = sum(self.bestscores)
        ratios = [f / total_scores for f in self.bestscores]
        return rd.choices(self.bestPilots, weights=ratios, k=2) # return a k-sized list 











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
        
            
            




# Atteinte d'un NN qui sait jouer est plus rapide sans weights=ratios

# Base : bestPilots, weights=ratios, [4,5,5,1], MR=0.1, STD=0.3    
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 29, best score: 217
# Generation 3, average score: 36, best score: 137
# Generation 4, average score: 43, best score: 217
# Generation 5, average score: 46, best score: 297
# Generation 6, average score: 58, best score: 463
# Generation 7, average score: 62, best score: 263
# Generation 8, average score: 85, best score: 427
# Generation 9, average score: 97, best score: 457
# Generation 10, average score: 86, best score: 465
# Generation 11, average score: 154, best score: 1020
# Generation 12, average score: 112, best score: 536
# Generation 13, average score: 168, best score: 899

# NN_LAYERS = [4, 10, 10, 1]
# Generation 1, average score: 28, best score: 126
# Generation 2, average score: 36, best score: 217
# Generation 3, average score: 39, best score: 137
# Generation 4, average score: 41, best score: 137
# Generation 5, average score: 37, best score: 263
# Generation 6, average score: 44, best score: 257
# Generation 7, average score: 42, best score: 223
# Generation 8, average score: 44, best score: 347
# Generation 9, average score: 48, best score: 267
# Generation 10, average score: 67, best score: 507
# Generation 11, average score: 89, best score: 1064
# Generation 12, average score: 81, best score: 823

# MR=0.05
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 29, best score: 187
# Generation 3, average score: 33, best score: 137
# Generation 4, average score: 43, best score: 217
# Generation 5, average score: 39, best score: 183
# Generation 6, average score: 49, best score: 220
# Generation 7, average score: 51, best score: 183
# Generation 8, average score: 51, best score: 457
# Generation 9, average score: 62, best score: 547
# Generation 10, average score: 71, best score: 903
# Generation 11, average score: 171, best score: 1386
# Generation 12, average score: 136, best score: 667

# MR=0.2
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 28, best score: 217
# Generation 3, average score: 34, best score: 177
# Generation 4, average score: 40, best score: 200
# Generation 5, average score: 43, best score: 385
# Generation 6, average score: 54, best score: 304
# Generation 7, average score: 62, best score: 340
# Generation 8, average score: 64, best score: 203
# Generation 9, average score: 60, best score: 263
# Generation 10, average score: 85, best score: 626
# Generation 11, average score: 75, best score: 1266
# Generation 12, average score: 92, best score: 987

# MR=0.5
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 27, best score: 186
# Generation 3, average score: 34, best score: 179
# Generation 4, average score: 38, best score: 186
# Generation 5, average score: 41, best score: 224
# Generation 6, average score: 47, best score: 586
# Generation 7, average score: 47, best score: 264
# Generation 8, average score: 50, best score: 225
# Generation 9, average score: 52, best score: 266
# Generation 10, average score: 73, best score: 703
# Generation 11, average score: 58, best score: 545

# MR=0.9
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 27, best score: 186
# Generation 3, average score: 32, best score: 146
# Generation 4, average score: 34, best score: 219
# Generation 5, average score: 38, best score: 504
# Generation 6, average score: 49, best score: 546
# Generation 7, average score: 57, best score: 257
# Generation 8, average score: 62, best score: 346
# Generation 9, average score: 56, best score: 426
# Generation 10, average score: 74, best score: 826
# Generation 11, average score: 91, best score: 426
# Generation 12, average score: 79, best score: 387

# std=0.1
# Generation 1, average score: 26, best score: 137
# Generation 2, average score: 29, best score: 187
# Generation 3, average score: 33, best score: 141
# Generation 4, average score: 43, best score: 217
# Generation 5, average score: 47, best score: 379
# Generation 6, average score: 58, best score: 305
# Generation 7, average score: 68, best score: 504
# Generation 8, average score: 66, best score: 185
# Generation 9, average score: 89, best score: 544
# Generation 10, average score: 86, best score: 1064
# Generation 11, average score: 163, best score: 1386
# Generation 12, average score: 222, best score: 2347
# Generation 13, average score: 233, best score: 1265
# Generation 14, average score: 287, best score: 2700