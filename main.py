import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D, ConvLSTM2D, Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random
import numpy as np
from collections import deque
from game import Game
import pygame
from PIL import Image
import threading
import time
from datetime import datetime
from keract import get_activations, display_activations
import cProfile
from google.colab import output

print("Modules initialised")

def buildDqnModel(inputShape, numActions):
    model = Sequential()

    # Convolutional layers
    model.add(ConvLSTM2D(16, (4, 4), strides=(2,2), activation='relu', input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (4, 4), strides=(2,2), activation='tanh'))
    model.add(BatchNormalization())

    # Reduce size for the fully connected layers to process
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(512, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(numActions, activation='linear'))

    return model

def epsilonGreedyPolicy(state, epsilon):
    if np.random.rand() < epsilon:
        v = random.randint(0, numActions - 1)
        return v  # Choose a random action
    else:
        QValues = mainModel.predict(state, verbose=0)
        return np.argmax(QValues[0])  # Choose the action with the highest Q-value

def pilImageToSurface(pilImage):
    return pygame.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()

drawQueue = deque()

def pygameWindow():
    screenRes = (256,224)

    pygame.init()
    window = pygame.display.set_mode((screenRes[0]*4, screenRes[1]*4))
    clock = pygame.time.Clock()

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        if len(drawQueue) > 0:
            im, episode = drawQueue.popleft()
            im = im.draw(episode=episode+1)
            im = im.resize((screenRes[0]*4, screenRes[1]*4), resample=Image.NEAREST)
            surface = pilImageToSurface(im)
            window.blit(surface, surface.get_rect())
        pygame.display.flip()
    pygame.quit()

def preprocess(x):
    return x

pygameThread = threading.Thread(target=pygameWindow)
pygameThread.start()

numStates = 3
inputShape = (numStates, 224, 256, 1)
numActions = 4
batchSize = 64
memorySize = 1000
numEpisodes = 100
epsilonStart = 1.0
epsilonEnd = 0.1
epsilonDecaySteps = 100
gamma = 0.99
targetUpdateFrequency = 20
saveFrequency = 15
stepsPerPrediction = 3

print("Building models")
mainModel = buildDqnModel(inputShape, numActions)
targetModel = buildDqnModel(inputShape, numActions)

optimiser = Adam(learning_rate=0.00025, epsilon=0.01)
mainModel.compile(optimiser, loss="mean_squared_error")
targetModel.compile(optimiser, loss="mean_squared_error")

targetModel.set_weights(mainModel.get_weights())
print("Models built")
print(mainModel.summary())

replayMemory = deque(maxlen=memorySize)

print("Starting training")
episodeRewards = []

try:
    for episode in range(numEpisodes):
        env = Game()
        state = env.drawToArray()
        episodeReward = 0
        episodeExperience = deque(maxlen=memorySize)  # Collect experience during the episode

        lastStates = []
        for i in range(numStates):
            lastStates.append(env.drawToArray())
        
        while True:
            epsilon = max(epsilonEnd, epsilonStart - episode / epsilonDecaySteps)
            action = epsilonGreedyPolicy(np.expand_dims(preprocess(np.array(lastStates)), (0,-1)), epsilon)

            reward = 0
            drawQueue.clear()
            for i in range(stepsPerPrediction-1):
                s, r, _ = env.MLstep(action)
                reward += r
                drawQueue.append((env, episode+1))
                lastStates.pop(0)
                lastStates.append(s)
            nextState, r, done = env.MLstep(action)
            reward += r
            drawQueue.append((env, episode+1))

            episodeReward += reward
            episodeExperience.append((lastStates, action, reward, lastStates[1:] + [nextState], done))
            state = nextState

            lastStates.pop(0)
            lastStates.append(state)

            output.clear(True, "status_text")
            with output.use_tags("status_text"):
                print(f"Episode: {episode}, Current reward: {episodeReward}", end="\r")

            if done:
                break

        # Update DQN model using collected experience after the episode ends
        if len(replayMemory) >= batchSize:
            minibatch = random.sample(replayMemory, batchSize)
            statesBatch, actionBatch, rewardBatch, nextStatesBatch, doneBatch = zip(*minibatch)

            QValues = mainModel.predict(np.expand_dims(preprocess(np.array(statesBatch)), -1), verbose=0)
            nextQValues = targetModel.predict(np.expand_dims(preprocess(np.array(nextStatesBatch)), -1), verbose=0)

            for i in range(batchSize):
                target = QValues[i].copy()

                if doneBatch[i]:
                    target[actionBatch[i]] = rewardBatch[i]
                else:
                    target[actionBatch[i]] = rewardBatch[i] + gamma * np.max(nextQValues[i])

                QValues[i] = target

            mainModel.fit(np.expand_dims(preprocess(statesBatch), -1), QValues)
        if episode % targetUpdateFrequency == 0:
            targetModel.set_weights(mainModel.get_weights())

        if episode % saveFrequency == 0:
            dtString = str(datetime.now()).replace(" ", "_").replace(":", "-")
            mainModel.save(f"space_invaders_dqn_main_{dtString}_checkpoint.keras")
            targetModel.save(f"space_invaders_dqn_target_{dtString}_checkpoint.keras")
            
        replayMemory += episodeExperience
        episodeRewards.append(episodeReward)
        print(f"Episode: {episode}, Total Reward: {episodeReward}, Epsilon: {epsilon}")

    ##        keractInputs = np.expand_dims(preprocess(np.array(lastStates)), (0))
    ##        activations = get_activations(mainModel, keractInputs)
    ##        display_activations(activations, cmap="gray", save=True, reshape_1d_layers=True, directory="layers/")
        
    dtString = str(datetime.now()).replace(" ", "_").replace(":", "-")
    mainModel.save(f"space_invaders_dqn_main_{dtString}.keras")
    targetModel.save(f"space_invaders_dqn_target_{dtString}.keras")
except:
    dtString = str(datetime.now()).replace(" ", "_").replace(":", "-")
    mainModel.save(f"space_invaders_dqn_main_{dtString}_RECOVERED.keras")
    targetModel.save(f"space_invaders_dqn_target_{dtString}_RECOVERED.keras")
