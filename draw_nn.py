import pygame
import numpy as np
import random as rd

WIDTH = 1200
HEIGHT = 600
NN_LAYERS = [4, 5, 5, 1]



def draw_nn(screen, network):
    screen.fill((255, 255, 255))

    # Nombre de couches et de neurones par couche
    num_layers = len(network) + 1
    layer_sizes = [network[0].shape[0]] + [w.shape[1] for w in network]

    # Positions des neurones
    x_spacing = 70 #WIDTH // (num_layers + 1)
    neuron_positions = []

    for i, layer_size in enumerate(layer_sizes):
        y_spacing = 150 // (layer_size + 1)
        neuron_positions.append([(900 + x_spacing * (i + 1), y_spacing * (j + 1)) for j in range(layer_size)])

    # Dessiner les connexions
    for i in range(len(network)):
        for j, neuron1 in enumerate(neuron_positions[i]):
            for k, neuron2 in enumerate(neuron_positions[i + 1]):
                weight = network[i][j, k]
                color = (255, 0, 0) if weight > 0 else (0, 0, 255)  # Rouge pour positif, bleu pour négatif
                thickness = int(abs(weight) * 3)  # Épaisseur proportionnelle au poids
                pygame.draw.line(screen, color, neuron1, neuron2, thickness)

    # Dessiner les neurones
    for layer in neuron_positions:
        for x, y in layer:
            pygame.draw.circle(screen, (0, 0, 0), (x, y), 5)  # Neurones en noir

    pygame.display.flip()



class Pilot():
   
    def __init__(self):
        self.initialize_weights()
        self.initialize_bias()
        
            
    def initialize_weights(self):
        self.weights = []
        for i in range(len(NN_LAYERS) - 1): 
            # Pour chaque couche du NN, creation d'une matrice de poids 
            layer = [[rd.uniform(-1, 1) for _ in range(NN_LAYERS[i+1])] for _ in range(NN_LAYERS[i])] # rd.gauss(0, 0.5)
            self.weights.append(np.array(layer))
            
        
    def initialize_bias(self):
        self.bias = []
        for layer in self.weights:
            nb_bias = np.size(layer, axis=1)
            self.bias.append(np.array([rd.uniform(-1, 1) for _ in range(nb_bias)])) # rd.gauss(0, 0.5)

nn = Pilot()


# Boucle principale pour afficher le réseau
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Réseau de Neurones")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_nn(screen, nn.weights)

pygame.quit()
