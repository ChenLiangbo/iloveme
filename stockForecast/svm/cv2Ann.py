import cv2
import numpy as np

c = 214750

nn_config = np.array([1250, 2, 1])

trainX = np.zeros((c, nn_config[0]))
trainY = np.zeros((c, nn_config[2]))

sampleWeights = np.ones_like(trainY) / trainY.shape[0]

ann = cv2.ANN_MLP(nn_config)
ann.train(inputs=trainX, outputs=trainY, sampleWeights=sampleWeights)