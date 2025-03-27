import tensorflow as tf
import pygad.kerasga
import numpy
import pygad

def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, keras_ga, model

    predictions = pygad.kerasga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)
    print(solution.shape)
    print(predictions.shape)
    print()

    mae = tf.keras.losses.MeanAbsoluteError()
    abs_error = mae(data_outputs, predictions).numpy() + 1e-8
    solution_fitness = 1.0 / abs_error

    return solution_fitness

def callback(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}\n")

# input_layer  = tf.keras.layers.Input([3])
# dense_layer1 = tf.keras.layers.Dense(5, activation="relu")(input_layer)
# output_layer = tf.keras.layers.Dense(1, activation="linear")(dense_layer1)
# model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

model = tf.keras.Sequential([
            tf.keras.layers.Input([3]),
            tf.keras.layers.Dense(5, activation="elu"),
            tf.keras.layers.Dense(1)
        ])

keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=5)

# Data inputs
data_inputs = numpy.array([[0.02, 0.1, 0.15],
                           [0.7, 0.6, 0.8],
                           [1.5, 1.2, 1.7],
                           [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = numpy.array([[0.1],
                            [0.6],
                            [1.3],
                            [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
ga_instance = pygad.GA(num_generations=250,
                       num_parents_mating=5, # Number of solutions to be selected as parents in the mating pool.
                       initial_population=keras_ga.population_weights,
                       fitness_func=fitness_func,
                       on_generation=callback)

ga_instance.run()



# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & Keras - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

# Make prediction based on the best solution.
predictions = pygad.kerasga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
print(f"Predictions : \n{predictions}")

mae = tf.keras.losses.MeanAbsoluteError()
abs_error = mae(data_outputs, predictions).numpy()
print(f"Absolute Error : {abs_error}")






# [ 0.21252628 -0.8047713  -0.7758761   0.54361425 -1.22238545 -1.65006587
#   0.25420808  0.02568321  0.09431894 -0.0394296  -0.42958849  1.13310023
#  -0.28462172  0.88488474 -0.10222979  0.70565694 -0.32910987 -0.83660664
#   0.03567408  0.79138019  1.2829606   0.6741969  -0.09694265  0.53592892
#   1.03602454  0.86929293] # (26,)

# [[2.3248487]
#  [1.04132  ]
#  [0.9611006]
#  [2.1619225]]