import random as rd
import pickle
import os
import matplotlib.pyplot as plt
from pilot import Pilot
from game import Session
from pathlib import Path
import copy as cp
import pygame as pg
pg.init()


    
POPULATION = 500
SURVIVAL_RATE = 0.1
N_GENERATIONS = 20

# Autres paramètres :
# nombre de layers NN
# fonctions activation 




class GeneticAlgo:
    

    def train(self):
        
        self.list_scores = []
        self.generation = 0
        
        self.population = [Pilot() for _ in range(POPULATION)]

        while self.generation < N_GENERATIONS:
            
            self.evaluate_generation()        
            self.bests_survives()
            self.change_generation()
            
            print(f"Generation {self.generation+1}, average score: {self.avgGenScore:.0f}, best score: {self.bestGenScore}")
            self.generation += 1
            
            # if self.ses.done:
            #     break
            
        self.evaluate_generation() # Evaluate the last generation
        self.bests_survives()
        self.bestPilotEver = self.bestPilots[-1]
        


    def evaluate_generation(self):
        self.scores = []
            
        ses = Session(POPULATION, self.generation)
        states = ses.reset()

        while not ses.done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    ses.done = True
            
            actions = [self.population[i].predict(states[i]) for i in range(len(self.population))]
            actions = [mat.tolist()[0][0] for mat in actions]
            states, self.scores = ses.step(actions)
            
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
        total_scores = sum(self.scores)
        ratios = [f / total_scores for f in self.scores]
        return rd.choices(self.bestPilots, k=2) # return a k-sized list 


    
    




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
        
            
            




# Attente d'un NN qui sait jouer est plus rapide sans weights=ratios




# choice dans population avec ratios
# Generation 1, average score: 25, best score: 177
# Generation 2, average score: 26, best score: 218
# Generation 3, average score: 30, best score: 306
# Generation 4, average score: 32, best score: 257
# Generation 5, average score: 40, best score: 861

# choice dans bestPilots avec ratios
# Generation 1, average score: 25, best score: 177
# Generation 2, average score: 31, best score: 144
# Generation 3, average score: 47, best score: 426
# Generation 4, average score: 60, best score: 586
# Generation 5, average score: 39, best score: 60
# Generation 6, average score: 70, best score: 457

# choice dans bestPilots sans ratios
# Generation 1, average score: 25, best score: 177
# Generation 2, average score: 30, best score: 946
# Generation 3, average score: 41, best score: 1667

