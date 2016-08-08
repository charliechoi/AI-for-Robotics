import os
from math import *
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
def readFile(filename):
    file = open(filename)
    data = [tuple([int(x[0]), int(x[1])]) for x in [line.replace('\n', '').split(',') for line in file.readlines()]]
    return data

def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return ((a + pi) % (pi * 2)) - pi

def save(pred, name):
    fout = open(name, 'w')
    fout.write(''.join(str(i[0])+','+str(i[1])+'\n' for i in pred))

def smooth(data):
    # smooth turning noise and distance in the hexbug
    nparray = np.array(data)
    x, y = nparray.T
    resampledx = savgol_filter(x,window_length=31,polyorder=3)
    resampledy = savgol_filter(y,window_length=31,polyorder=3)
    smoothed = np.column_stack((resampledx, resampledy)).tolist()
    return smoothed

def adjust_to_collision(collision, r):
    try:
        if collision[0] == 1:
            r.heading = pi - r.heading
        elif collision[0] == 2:
            r.heading = -r.heading
        elif collision[0] == 3:
            r.heading = pi - r.heading
        elif collision[0] == 4:
            r.heading = -r.heading
        elif collision[0] == 4:
            r.heading = -r.heading
        elif collision[0][0] ==5:
            circ_angle = atan2((collision[0][2]-175), (collision[0][1]-320))
            r.heading = angle_trunc(pi-r.heading-2*circ_angle)
    except TypeError:
        pass
    return r

def get_box_dim(data):
    maxX = -100;maxY = -100;minX = 10000;minY = 10000

    # finding the corner coordinates by finding extremes from raw data
    for i in range(len(data)):
        maxX = max(int(data[i][0]), maxX)
        maxY = max(int(data[i][1]), maxY)
        minX = min(int(data[i][0]), minX)
        minY = min(int(data[i][1]), minY)

    # find box size
    bw = maxX - minX
    bh = maxY - minY

    return maxX, maxY, minX, minY, bw, bh

def get_state_difference(xi, yi, xf, yf):
    # heading
    h = angle_trunc(atan2((yf - yi), (xf - xi)))

    # distance
    d = sqrt((yf - yi) ** 2 + (xf - xi) ** 2)

    return h, d

def measurement_prob(measurement, prediction):

    distance = sqrt((measurement[0] - prediction[0])**2+(measurement[1]-prediction[1])**2)

    # update Gaussian
    distance_weight = float(exp(-(distance ** 2) / (20 ** 2) / 2.0) /sqrt(2.0 * pi * (20 ** 2)))

    error = distance_weight
    return error

def heading_prob(measured, predicted):
    heading = abs(measured-predicted)
    heading_weight = (exp(- (heading ** 2) / (0.05** 2) / 2.0) /sqrt(2.0 * pi * (0.05 ** 2)))
    return heading_weight

def turning_prob(measured, predicted):
    turning = abs(measured - predicted)
    turning_weight = (exp(- (turning** 2) / (0.5 ** 2) / 2.0) / sqrt(2.0 * pi * (0.5 ** 2)))
    return turning_weight

def learn(training_data):
    pos_turning = []
    neg_turning = []
    n_pos =0
    n_neg=0
    distance = 0
    s0 = training_data[0]
    s1 = training_data[1]
    h, d = get_state_difference(s0[0], s0[1], s1[0], s1[1])
    for i in range(2, len(training_data)):
        h_old = h
        curr = training_data[i]
        prev = training_data[i - 1]
        h, d = get_state_difference(prev[0], prev[1], curr[0], curr[1])
        distance += d
        if h>0:
            pos_turning.append( h - h_old)
            n_pos+=1
        else:
            neg_turning.append(h-h_old)
            n_neg+=1
        #turning += h - h_old
    distance /= (len(training_data) - 2)
    pos_turning=np.median(pos_turning)
    neg_turning=np.median(neg_turning)
    #turning /= (len(training_data) - 2)
    return distance, pos_turning, neg_turning


def get_initial_states(data):
    # Get initial position and heading of the hexbug after three motions

    s1 = data[0]
    s2 = data[1]
    s3 = data[2]

    # get initial position after 3 points
    xi = s3[0];yi = s3[1]

    # get initial states
    h1, d1 = get_state_difference(s1[0], s1[1], s2[0], s2[1])
    h2, d2 = get_state_difference(s2[0], s2[1], s3[0], s3[1])
    turn = h2 - h1
    di = (d1 + d2) / 2

    return xi, yi, h2
def get_average_properties(particles):
    x = 0;y = 0;heading = 0;distance = 0;turning = 0
    N = len(particles)
    for i in range(N):
        r = particles[i]
        x += r.x
        y += r.y
        heading += r.heading
        distance += r.distance
        turning += r.turning
    x /= N
    y /= N
    heading /= N
    distance /= N
    turning /= N
    return [x, y, heading, distance, turning]

def error(l1, l2):
    return sum((c - a)**2 + (d - b)**2 for ((a, b), (c, d)) in zip(l1, l2))**0.5
#test readFile
#data = readFile('Final_Project_Resources-2016-07-16'+os.sep+'Final_Project_Resources'+os.sep+'inputs-txt'+os.sep+'test01.txt')
#print data
#save([[1,2],[2,3]])
#x = -2*pi
#print(angle_trunc(x))
