from particle_filter import *
import argparse
from utils import *
import os
import time

t = time.time()
parser = argparse.ArgumentParser(description="Final Project CS8803")
parser.add_argument('--test', '-t', help='test file')

args = parser.parse_args()
dir_path = os.path.dirname(os.path.realpath(__file__))
training = readFile('training_data.txt')
avgd, avg_pos_t, avg_neg_t = learn(training)
file = readFile(os.path.join(dir_path,args.test))

p = ParticleFilter(file, avgd, 0)
train_log=p.train()
predictions = p.predict_next_move(avg_pos_t, avg_neg_t)

save(predictions, 'prediction.txt')
print (time.time()-t)