import random as rd
import numpy as np
import random as rd
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf

# SEED = 42
# rd.seed(SEED)
# tf.random.set_seed(SEED)




# model = tf.keras.Sequential([
#             tf.keras.layers.Input([3]),
#             tf.keras.layers.Dense(5, activation="elu"),
#             tf.keras.layers.Dense(1)
#         ])


# # print(model.weights[0].numpy())
# # print(model.count_params())
# print(model.summary())

# # for i in range(len(model.weights)):
# #     print(model.weights[i].numpy().shape)
# # print()




# print("start")
# if 0.00:
#     print("true")




# class ReplayMemory:
#     def __init__(self, capacity):
#         self.memory = deque(maxlen=capacity)
    
#     def push(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
    
#     def sample(self):
#         samples = rd.sample(self.memory, 32)
#         return zip(*samples)
    
#     def __len__(self):
#         return len(self.memory)
    
    
# memory = ReplayMemory(100)
# for i in range(60):
#     memory.push(i, i, i, i, i)
    

# batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample()

# print(batch_state)
    
    

# replay_buffer = deque(maxlen=2000)

# def sample_experiences():
#     indices = np.random.randint(len(replay_buffer), size=32)
#     batch = [replay_buffer[index] for index in indices]
#     return [[experience[field_index] for experience in batch] for field_index in range(5)] 


# for i in range(10):
#     replay_buffer.append((i, i, i, i, i))
    
# print(sample_experiences())



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
        

nn = Neural_net()

print(nn.model.predict(np.array([0.1, 0.2, 0.3, 0.4]).reshape(1, 4), verbose=100)[0])