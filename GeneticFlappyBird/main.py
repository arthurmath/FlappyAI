import pygame
import random
from learning import *
pygame.init()

SCALE = 1
WIDTH = SCALE * 800  # Largeur de l'écran
HEIGHT = SCALE * 600  # Hauteur de l'écran

screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

birds = []
pipes = []
pipe_timer = 1500
front_pipe = None
dead_birds = []
gen = 1

class Bird:
    def __init__(self):
        global birds
        birds.append(self)
        self.score = 0
        self.color = (0, 0, 0)
        self.y = 0.5
        self.x = 0
        self.y_velocity = 0
        self.brain = Net()
        self.gravity = 0.0003
        self.dead = False

    def update(self):
        global front_pipe, dead_birds
        if not self.dead:
            self.score += 1
        self.y_velocity += self.gravity
        self.y += self.y_velocity

        if self.brain.predict([self.y, self.y_velocity, front_pipe.height, front_pipe.x]):
            self.y_velocity = -0.008

        if self.y > 1 or self.y < 0:
            if not self.dead:
                dead_birds.append(self)
            self.dead = True
            return False
        
        for p in pipes:
            if self.x > p.x and self.x < p.x + 0.1 and abs(self.y - p.height) > 0.1:
                if not self.dead:
                    dead_birds.append(self)
                self.dead = True
                return False
        return True

    def draw(self):
        if not self.dead:
            pygame.draw.circle(screen, self.color, (WIDTH // 2, int(self.y * HEIGHT)), int(HEIGHT * 0.025))

class Pipe:
    def __init__(self):
        global pipes
        pipes.append(self)
        self.x = 1
        self.color = (0, 0, 0)
        self.height = random.uniform(0.125, 0.875)
    
    def update(self):
        self.x -= 0.01
        if self.x < -0.1:
            pipes.pop(0)
    
    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x * HEIGHT, 0, HEIGHT * 0.1, HEIGHT * (self.height - 0.1)))
        pygame.draw.rect(screen, self.color, (self.x * HEIGHT, (self.height + 0.1) * HEIGHT, HEIGHT * 0.1, HEIGHT))

def update():
    global pipe_timer, front_pipe, gen, birds, pipes, dead_birds
    
    screen.fill((255, 255, 255))
    
    if len(dead_birds) == len(birds):
        gen += 1
        birds.sort(key=lambda b: b.score, reverse=True)
        parents = birds[:10]
        
        for parent in parents:
            parent.dead = False
            parent.y = 0.5
            parent.y_velocity = 0
            parent.score = 0
        
        birds = []
        dead_birds = []
        pipes = []
        
        for _ in range(999):
            new_bird = Bird()
            new_bird.brain = mate(random.choice(parents).brain, random.choice(parents).brain)
        
        birds.append(parents[0])
    
    pipe_timer += 1000 / 60
    if pipe_timer > 1500:
        pipe_timer = 0
        Pipe()
    
    for pipe in pipes:
        if pipe.x > -0.1:
            front_pipe = pipe
            break
    
    for bird in birds:
        bird.update()
        bird.draw()
    
    for pipe in pipes[:]:
        pipe.update()
        pipe.draw()
    
    pygame.display.flip()

def main():
    global birds
    for _ in range(1000):
        Bird()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        update()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()
