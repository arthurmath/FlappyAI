import os
import time
import pickle
import numpy as np
import random as rd
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from game import Session

from collections import deque
replay_buffer = deque(maxlen=2000)



SEED = 42
LR = 1e-2
N_STEPS = 200
BATCH_SIZE = 16 # 32
N_EPISODES = 100 # 600
POPULATION = 10
EPS_FACTOR = int(N_EPISODES * 5 / 6) 
DISCOUNT_FACTOR = 0.95
WEIGHTS_PATH = Path() / "weights"
IMAGES_PATH = Path() / "results_dqn/images"

rd.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)





    
class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=int(1e4))
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self):
        samples = rd.sample(self.memory, 32)
        return zip(*samples)
    
    def __len__(self):
        return len(self.memory)
    
    
    
class Neural_net:
    def __init__(self):
        
        self.input_shape = [4]
        self.output_shape = 1 

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(self.input_shape),
            tf.keras.layers.Dense(8, activation="elu"),
            tf.keras.layers.Dense(8, activation="elu"),
            tf.keras.layers.Dense(self.output_shape, activation="relu")
        ])




class DeepQN:
    
    def __init__(self):
        self.memory = ReplayMemory()
        self.env = Session(POPULATION)
        self.population = [Neural_net() for _ in range(POPULATION)]
        
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=LR)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        

    def train(self):
        
        self.list_bests = []
        self.list_avg = []
        self.best_score_all = 0

        for self.generation in range(N_EPISODES):
            
            self.evaluate_generation()
            
            self.update_scores()

            if self.generation > int(BATCH_SIZE * 1.5):
                self.training_step()
                
            print(f"Episode {self.generation+1}, avg score: {self.avg_score:.0f}, best score: {self.best_score}, done {self.end_step}")
            
    
    
    
    def evaluate_generation(self):
            
        self.epsilon = max(1 - self.generation / EPS_FACTOR, 0.01)
        self.ses = Session(POPULATION, self.generation)
        self.states = self.ses.reset()
        self.end_step = 0

        while not self.ses.done:
            actions = self.epsilon_greedy_policy()
            
            next_states, self.scores, dones = self.ses.step(actions)
            
            self.memory.push(self.states, actions, self.scores, next_states, dones)
            self.end_step += 1
            
        
    def epsilon_greedy_policy(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.population[0].output_shape+1, size=POPULATION)  # random action 
        else:
            Q_values = [self.population[i].model.predict(np.array(self.states[i]).reshape(1, 4), verbose=0)[0] for i in range(len(self.population))]
            res = [1 if val > 0 else 0 for val in Q_values]
            return res
            

            
    def update_scores(self):
        self.best_score = max(self.scores)
        self.avg_score = sum(self.scores) / POPULATION
        self.list_bests.append(self.best_score)
        self.list_avg.append(self.avg_score)
        
        # if self.best_score >= self.best_score_all:
        #     self.best_weights = self.model.get_weights()
        #     self.best_score_all = self.best_score
    
    

    def training_step(self):
        
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        next_Q_values = self.model.predict(np.array(next_states), verbose=0)
        max_next_Q_values = next_Q_values.max(axis=1)
        runs = np.ones(len(dones)) - np.array(dones)
        target_Q_values = rewards + runs * DISCOUNT_FACTOR * max_next_Q_values
        target_Q_values = target_Q_values.reshape(-1, 1)
        
        mask = tf.one_hot(actions, self.output_shape)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(np.array(states))
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  
        
        

        
        
        
        
        
    
def main():
    
    train = True
    dqn = DeepQN()
    n_train = len(os.listdir(WEIGHTS_PATH)) # nb de fichiers dans dossier weights
    
    if train:
        # Warm start
        # with open(WEIGHTS_PATH / Path(f"colab1.weights"), "rb") as f:
        #     weights = pickle.load(f)
        #     dqn.model.set_weights(weights)
        
        start = time.time()
        dqn.train()
        print(f"\nDurée entrainement avec {N_EPISODES} épisodes : {(time.time() - start)/60}min\n")
        
        with open(WEIGHTS_PATH / Path(f"{n_train}.weights"), "wb") as f:
            pickle.dump((dqn.best_weights), f)
        
        plt.figure(figsize=(8, 4))
        plt.plot(dqn.list_avg, label='Average scores')
        plt.plot(dqn.list_bests, label='Best scores')
        plt.xlabel("Générations")
        plt.ylabel("Scores (%)")
        plt.grid(True)
        plt.legend()
        plt.show()

        
    else:
        with open(WEIGHTS_PATH / Path(f"colab3.weights"), "rb") as f:
            weights = pickle.load(f)
        
        dqn.model.set_weights(weights)
        
        env = Session(display=True)
        state = env.reset()
        done = False

        while not done:
            moves = dqn.model.predict(np.array(state)[np.newaxis, :], verbose=0)[0].argmax()
            state, _, done = env.step(moves)
            done = False
    
    dqn.env.close()





if __name__=='__main__':
    main()
    


    
    




        



# Impossible de faire plusieurs actions en meme temps


# next_state = self.normalize_state(next_state)
# next_state.append(self.previous_moves) # #######

# prendre le espilon exponentiel de gpt.py


