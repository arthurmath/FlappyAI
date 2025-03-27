import numpy as np
import pygad
import pygad.nn
import pygad.gann
from game import Session


SEED = 42
BATCH = 5
POPULATION = 2 * BATCH
SURVIVAL_RATE = 0.1
N_GENERATIONS = 1
N_STEPS = 100    
EPISODE_INCREASE = 2
NN_LAYERS = [4, 3, 3, 1]



# Ne peut pas marcher car data_inputs est constant
    

def fitness_function(ga_instance, solution, nn_idx):
    """ Fonction de coût qui évalue la performance d'un réseau de neurones. """
    global gann
        
    ses = Session(nb_pilots=len(nn_idx), generation=ga_instance.generations_completed)
    states = ses.reset()

    while not ses.done:
        print("LEN : ", len(nn_idx))
        print(nn_idx)
        print("STATES : ", states, len(states))
        actions = [pygad.nn.predict(last_layer=gann.population_networks[nn], data_inputs=np.array(states[i]).reshape(1, 4)) for i, nn in enumerate(nn_idx)]
        print("ACTIONS :", actions, "\n\n")
        # actions = [[0]] * len(nn_idx)
        states, scores = ses.step(np.array(actions).reshape(len(actions)))
        
    return scores



def callback(ga_instance): 
    global gann
    population_matrices = pygad.gann.population_as_matrices(population_networks=gann.population_networks, population_vectors=ga_instance.population)
    gann.update_population_trained_weights(population_trained_weights=population_matrices)
    print(f"Generation: {ga_instance.generations_completed}, Best fitness: {ga_instance.best_solution()[1]}")





# Création de la population initiale
gann = pygad.gann.GANN(num_solutions=POPULATION,
                       num_neurons_input=NN_LAYERS[0],
                       num_neurons_hidden_layers=NN_LAYERS[1:-1],
                       num_neurons_output=NN_LAYERS[-1], 
                       hidden_activations="relu",
                       output_activation="softmax",
                       )


# print(dir(gann))


# for i, nn in enumerate(gann.population_networks):
#     print(f"NN {i} Weights: {nn}\n")








initial_pop = pygad.gann.population_as_vectors(population_networks=gann.population_networks)


# Définition des paramètres de l'algorithme génétique
ga_instance = pygad.GA(num_generations = N_GENERATIONS,
                    num_parents_mating = 10,
                    initial_population = initial_pop,
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