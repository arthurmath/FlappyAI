import random as rd

MUTATION_RATE = 0.01

class Neuron:
    def __init__(self, nb_inputs):
        self.weights = [rd.uniform(-1, 1) for _ in range(nb_inputs+1)]
    
    def predict(self, inputs):
        
        total = sum(w * i for w, i in zip(self.weights[:-1], inputs))
        total += self.weights[-1]
        
        return 1 if total > 0 else 0

class Layer:
    def __init__(self, nb_inputs, num_neurons):
        self.neurons = [Neuron(nb_inputs) for _ in range(num_neurons)]

class Net:
    def __init__(self):
        self.layers = []
        self.layers.append(Layer(4, 3))
        self.layers.append(Layer(3, 3))
        self.layers.append(Layer(3, 1))
    
    def predict(self, vector):
        
        for layer in self.layers:
            vector = [neuron.predict(vector) for neuron in layer.neurons]
        
        return vector[0]
    


def mate(parent1, parent2):
    child = Net()
    
    for i in range(len(parent1.layers)):
        for j in range(len(parent1.layers[i].neurons)):
            for k in range(len(parent1.layers[i].neurons[j].weights)):
                if rd.random() > 0.5:
                    child.layers[i].neurons[j].weights[k] = parent1.layers[i].neurons[j].weights[k]
                else:
                    child.layers[i].neurons[j].weights[k] = parent2.layers[i].neurons[j].weights[k]
                
                if rd.random() < MUTATION_RATE:
                    child.layers[i].neurons[j].weights[k] = rd.uniform(-1, 1)
    
    return child
