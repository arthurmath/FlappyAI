import numpy as np
import pygad
import pygad.nn
import pygad.gann
import pygad.kerasga
from game import Session
import tensorflow as tf


SEED = 42
BATCH = 5
POPULATION = 2 * BATCH
SURVIVAL_RATE = 0.1
N_GENERATIONS = 1
N_STEPS = 100    
EPISODE_INCREASE = 2
NN_LAYERS = [4, 3, 3, 1]


    

def fitness_function(ga_instance, solution, nn_idx):
    """ Fonction de coût qui évalue la performance d'un réseau de neurones. """
    global gann, states

    while not ses.done:
        actions = [pygad.kerasga.predict(model=model, solution=gann.population_weights[nn], data=np.array(states[i]).reshape(1, 4)) for i, nn in enumerate(nn_idx)]
        actions = [1 if act > 0 else 0 for act in actions] # [0, 1, 1, 1, 1] 
        states, scores = ses.step(actions)
        
        print("ACTIONS :", actions, "\n\n")
    return scores



def callback(ga_instance):
    states = ses.reset()
    print(f"Generation: {ga_instance.generations_completed}, Best fitness: {ga_instance.best_solution()[1]}")



model = tf.keras.Sequential([
            tf.keras.layers.Input([4]),
            tf.keras.layers.Dense(3, activation="elu"),
            tf.keras.layers.Dense(3, activation="elu"),
            tf.keras.layers.Dense(1)
        ])

gann = pygad.kerasga.KerasGA(model=model, num_solutions=POPULATION)

ses = Session(nb_pilots=BATCH)
states = ses.reset()


# Définition des paramètres de l'algorithme génétique
ga_instance = pygad.GA(num_generations = N_GENERATIONS,
                    num_parents_mating = 10,
                    initial_population = gann.population_weights,
                    # sol_per_pop = POPULATION,
                    # num_genes = X,
                    fitness_func = fitness_function,
                    mutation_percent_genes = 10,
                    crossover_type = 'single_point',
                    mutation_type = 'random',
                    parent_selection_type = 'tournament',
                    keep_parents = 2, 
                    init_range_high = 1,
                    init_range_low = -1,
                    random_seed = SEED,
                    fitness_batch_size = BATCH,
                    # parallel_processing = 5,
                    on_generation=callback,
                    )

# Exécution de l'algorithme génétique
ga_instance.run()

# Récupération du meilleur individu
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Meilleure solution: {solution}")
print(f"Fitness: {solution_fitness}")








# # pourquoi nn_idx passe de len 32 à 31 au bout de 3 steps ? 
# # Quel est le lien entre population et batch ? 
# # Est ce que update_population_trained_weights est nécessaire ? 