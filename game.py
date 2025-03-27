import random
import numpy as np
from pilot import Pilot
from pathlib import Path
import pickle
import sys
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg


SEED = 42
WIDTH = 1200
HEIGHT = 600
WHITE = (255, 255, 255)
FPS = 100 
GRAVITY = 2      # Accélération due à la gravité
JUMP_STRENGTH = -15  # Vélocité initiale du saut (négatif car va vers le haut)
MAX_SPEED = 120  # Vitesse max de chute

random.seed(SEED)




class Bird:
    
    bird_velocity = 10

    def __init__(self, ses):
        self.ses = ses
        self.bird_img = self.ses.bird_img
        
        self.bird_img_rect = self.bird_img.get_rect()
        self.bird_img_rect.left = 200
        self.bird_img_rect.top = HEIGHT/2 - 50
        self.bird_img_rect.height = 35
        self.bird_img_rect.width = 45
        
        self.speed = JUMP_STRENGTH  # Vitesse verticale
        self.letgo = True  # Empêche le spam de saut
        self.alive = True

    def update(self, action):
        if action:
            self.speed = JUMP_STRENGTH
                
        if self.bird_img_rect.top <= 0:
            self.alive = False
        if self.bird_img_rect.bottom >= HEIGHT - 80:
            self.alive = False
        
        self.speed += GRAVITY
        self.speed = min(self.speed, MAX_SPEED)

        self.bird_img_rect.top += self.speed
        
        for pipe in self.ses.pipes:
            if self.bird_img_rect.right > pipe.pipe_img_rect.left - 10:
                if pipe.pipe_img_rect.colliderect(self.bird_img_rect) or pipe.flipped_pipe_rect.colliderect(self.bird_img_rect):
                    self.alive = False
            
    def draw(self):
        self.ses.screen.blit(self.bird_img, self.bird_img_rect)
        # pg.draw.rect(self.ses.screen, (255, 0, 0), self.bird_img_rect, 2) # heatbox
        
    def check_event(self, event):
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RSHIFT and self.letgo:
                self.velocity = self.JUMP_STRENGTH  # Applique l'impulsion du saut
                self.letgo = False  # Bloque tant que la touche n'est pas relâchée
        if event.type == pg.KEYUP:
            if event.key == pg.K_RSHIFT:
                self.letgo = True  # Permet un nouveau saut
                
    def start(self):
        self.start_rect = self.bird_img.get_rect(center=(220, HEIGHT/2 - 50)) 
        self.ses.screen.blit(self.bird_img, self.start_rect)
            
    @classmethod
    def speedup(cls):
        cls.bird_velocity += 1
        
    @classmethod
    def restart(cls):
        cls.bird_velocity = 10
        





class Pipe:
    pipe_velocity = 10
    add_new_pipe_rate = 40  

    def __init__(self, ses):
        self.ses = ses
        # self.hole = random.randint(-30, 200) # old easy
        self.hole = random.randint(-40, 200) 
        
        self.pipe_img = self.ses.pipe_img
        self.flipped_pipe = ses.flipped_pipe
        
        self.pipe_img_rect = self.pipe_img.get_rect()
        self.pipe_img_rect.top = HEIGHT/2 + self.hole
        self.pipe_img_rect.right = WIDTH + 70
        self.flipped_pipe_rect = self.flipped_pipe.get_rect()
        # self.flipped_pipe_rect.top = HEIGHT/2 - 700 + self.hole # old easy
        self.flipped_pipe_rect.top = HEIGHT/2 - 650 + self.hole
        self.flipped_pipe_rect.right = WIDTH + 70
        
    def update(self):
        if self.ses.loop_counter % self.add_new_pipe_rate == 0:
            self.ses.add_pipe = True
        
        # Fait beuger le mouvement des pipes 
        if self.pipe_img_rect.left <= -70:
            self.ses.pipes.pop(0)
        else:
            self.pipe_img_rect.right -= self.pipe_velocity
            self.flipped_pipe_rect.right -= self.pipe_velocity
                       
                    
    def draw(self):
        self.ses.screen.blit(self.pipe_img, self.pipe_img_rect)
        self.ses.screen.blit(self.flipped_pipe, self.flipped_pipe_rect)
        # pg.draw.rect(self.ses.screen, (255, 0, 0), p.pipe_img_rect, 2) # heatbox
        # pg.draw.rect(self.ses.screen, (255, 0, 0), p.flipped_pipe_rect, 2) # heatbox




class Background:
    def __init__(self, ses):
        self.ses = ses
        self.x = -5
        self.y = -120
        self.speed = 6
        self.image = self.ses.background_img

    def update(self):
        self.x = (self.x - self.speed) % -WIDTH

    def draw(self):
        self.ses.screen.blit(self.image, (self.x, self.y))
        self.ses.screen.blit(self.image, (WIDTH + self.x, self.y))
        


class Ground:
    def __init__(self, ses, pipe):
        self.ses = ses
        self.pipe = pipe[0]
        self.x = -5
        self.y = 550
        self.speed = 6
        self.image = self.ses.ground_img

    def update(self):
        self.x = (self.x - self.pipe.pipe_velocity) % -WIDTH

    def draw(self):
        self.ses.screen.blit(self.image, (self.x, self.y))
        self.ses.screen.blit(self.image, (WIDTH + self.x + 10, self.y))


        
class Score:
    def __init__(self, ses):
        self.ses = ses
        self.current = 0
        self.score_rate = 70
        
    def update(self):
        if self.ses.loop_counter % self.score_rate == 0:
            self.current += 1
        if self.current > self.ses.highscore:
            self.ses.highscore = self.current
        
    def draw(self):
        self.score_font = self.ses.text.render(f"Score: {self.current} ", True, WHITE)
        self.high_font = self.ses.text.render(f"Generation: {self.ses.generation+1}", True, WHITE) 
        self.speed_font = self.ses.text.render(f"Population: {self.ses.nb_alive}", True, WHITE) 
        self.ses.screen.blit(self.score_font, (50, 20))
        self.ses.screen.blit(self.high_font, (200, 20))
        self.ses.screen.blit(self.speed_font, (400, 20))
        




class Session:
    def __init__(self, nb_pilots=10, display=True):
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption('Flappy Bird')
        self.title = pg.font.SysFont('forte', 60)
        self.text = pg.font.SysFont('forte', 30)
        self.highscore = 0
        self.display = display
        self.nb_pilots = nb_pilots
        
        self.load_images()
        self.generate_objects()
         
    def load_images(self):
        bird_img = pg.image.load('images/bird.png').convert_alpha()
        img_width, img_height = bird_img.get_size()
        self.bird_img = pg.transform.scale(bird_img, (img_width//18, img_height//18))
        
        self.pipe_img = pg.image.load('images/pipe2.png').convert_alpha()
        img_width, img_height = self.pipe_img.get_size()
        self.pipe_img = pg.transform.scale(self.pipe_img, (img_width // 2.5, img_height // 2.5))
        self.flipped_pipe = pg.transform.flip(self.pipe_img, False, True) 
        
        self.background_img = pg.image.load('images/background.png').convert()
        self.ground_img = pg.image.load('images/ground.png').convert()

    def generate_objects(self):
        self.bird_list = [Bird(self) for _ in range(self.nb_pilots)]
        self.background = Background(self)
        self.pipes = [Pipe(self)] 
        self.ground = Ground(self, self.pipes)
        self.score = Score(self)
    
    def reset(self, nn=None, generation=1):
        self.states = []
        self.scores = [0] * self.nb_pilots
        self.nb_alive = self.nb_pilots
        self.generation = generation
        next_pipe = self.pipes[0]
        self.level_up_rate = 250
        self.loop_counter = 0
        self.add_pipe = False
        self.done = False
        self.quit = False
        self.level = 1
        if nn is not None:
            self.best_nn = nn
        for bird in self.bird_list:
            self.states.append([bird.bird_img_rect.top, bird.speed, next_pipe.hole, next_pipe.pipe_img_rect.right])   
        self.normalisation() 
        return self.states
        
    def update(self, actions):
        self.background.update()
        for idx, bird in enumerate(self.bird_list):
            bird.update(actions[idx])
        for pipe in self.pipes:
            pipe.update()
        if self.add_pipe:
            self.pipes.append(Pipe(self))
            self.add_pipe = False
        self.nb_alive = sum([bird.alive for bird in self.bird_list])
        self.score.update()
        self.ground.update()
        self.clock.tick(FPS)
        
    def draw(self):
        self.background.draw()
        for bird in self.bird_list:
            if bird.alive:
                bird.draw()
        for pipe in self.pipes:
            pipe.draw()
        self.score.draw()
        self.ground.draw()
        if 'self.best_nn' in locals():
            self.draw_nn(self.best_nn)
        pg.display.flip()


    def step(self, actions):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
                self.quit = True
        
        self.loop_counter += 1
        self.update(actions)
        if self.display:
            self.draw()
        
        dones = [bird.alive for bird in self.bird_list]
        if not any(dones):
            self.done = True
        
        if self.bird_list[0].bird_img_rect.left < self.pipes[0].pipe_img_rect.left:
            next_pipe = self.pipes[0]
        else:
            next_pipe = self.pipes[1]
            
        for i, bird in enumerate(self.bird_list):
            if bird.alive:
                self.states[i] = [bird.bird_img_rect.top, bird.speed, next_pipe.hole, next_pipe.pipe_img_rect.left]
                self.scores[i] += 1
                
        self.normalisation()
        
        return self.states, self.scores, dones


    def normalisation(self):
        """ Il faut que les entrées soient dans [-1, 1] pour converger """
        list_ranges = [[0, 600], [-MAX_SPEED/2, MAX_SPEED], [-30, 200], [200, 1200]]
        for i, state in enumerate(self.states):
            self.states[i] = [self.scale(np.array(state[i]), *list_ranges[i]) for i in range(len(state))]
        
    def scale(self, x, a, b):
        """Transforme la valeur x initialement comprise dans l'intervalle [a, b]
            en une valeur comprise dans l'intervalle [-1, 1]."""
        return 2 * (x - a) / (b - a) - 1
    
    
    def draw_nn(self, network):
        network = network.weights

        # Nombre de neurones par couche
        layer_sizes = [network[0].shape[0]] + [w.shape[1] for w in network]

        # Positions des neurones
        x_spacing = 70 
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
                    pg.draw.line(self.screen, color, neuron1, neuron2, thickness)

        # Dessiner les neurones
        for layer in neuron_positions:
            for x, y in layer:
                pg.draw.circle(self.screen, WHITE, (x, y), 5)  # Neurones en noir

    
    def close(self):
        pg.quit()







if __name__ == '__main__':
    
    nb_birds = 1
    ses = Session(nb_birds)
    states = ses.reset()
    
    while not ses.done:
        ### RANDOM ###
        # proba = 0.9
        # actions = np.random.choice(2, p=[proba, 1-proba], size=nb_birds)
        #################
        
        ### HUMAN ###
        actions = [0]
        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_RSHIFT:
                    actions = [1]
            if event.type == pg.KEYUP:
                if event.key == pg.K_RSHIFT:
                    actions = [0]
        #################
        
        ### SAVED WEIGHTS ###
        # n_train = len(os.listdir(Path("weights"))) # nb de fichiers dans dossier weights
        # with open(Path("weights") / Path(f"{n_train-1}.weights"), "rb") as f:
        #     weights, bias = pickle.load(f)
        #     agent = Pilot(weights, bias)
        # actions = [agent.predict(states).tolist()[0][0]]
        #################
        
        states, scores, _ = ses.step(actions)
        # print([round(x, 2) for x in states[0]])
    
    print(sorted(scores))
    pg.quit()
    sys.exit(0)
    
    








