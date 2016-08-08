import numpy as np
from math import *
from utils import *
from robot import *
import random
import os
import time

class ParticleFilter:

    particle_size = 150
    hexbug_length = 36
    no_coll = 0
    left_coll = 1
    bottom_coll=2
    right_coll=3
    top_coll=4
    candle = 5
    candle_center_x = 320
    candle_center_y = 175
    radius = 29

    def __init__(self, training_data, di, ti):
        #print training_data
        self.training = smooth(training_data)
        #self.training = training_data
        xi,yi,self.hi= get_initial_states(training_data)
        #print xi, yi, hi, di, turn
        self.maxX, self.maxY, self.minX, self.minY, self.bh, self.bw = get_box_dim(training_data)
        self.particles = self.init_particles(self.particle_size, di, ti, 0.05, 1, 0.05)
        #print self.maxX, self.maxY, self.minX, self.minY, self.bh, self.bw
        self.collision = self.get_collisions(training_data)
        self.latest_data = None


    def init_particles(self,N, distance, turning, t_noise, m_noise, d_noise):
        #Initialize particles randomly within the box's dimension

        particles = []

        for i in range(N):
            x_init = random.random() * self.maxX
            y_init = random.random() * self.maxY
            h_init = random.random() * 2*pi
            d_init = random.gauss(distance, d_noise)
            t_init = angle_trunc(random.gauss(turning, t_noise))
            r = robot(x=x_init, y=y_init, heading=h_init, turning=t_init,distance=d_init)
            r.set_noise(t_noise, d_noise, m_noise)
            particles.append(r)

        #print "particles made"
        return particles

    def pcopy(self, particles):
        #Helper function to copy robot class object
        copied = []
        for i in range(len(particles)):
            particle = particles[i]
            r = robot(particle.x, particle.y, particle.heading, particle.turning, particle.distance)
            r.set_noise(particle.turning_noise, particle.distance_noise, particle.measurement_noise)
            copied.append(r)

        return copied

    def train(self):
        #Update the particle fleet every new line of data.
        train_log =[]
        N = len(self.particles)
        trueh = self.hi
        for t in range(2,len(self.training)):
            new_data = self.training[t]
            old_data = self.training[t-1]
            old_trueh = trueh
            trueh, trued = get_state_difference(old_data[0], old_data[1], new_data[0], new_data[1])
            #print new_data
            new_robots=[]
            collision = self.collision[t-3]
            #print collision

            for k in range(0,len(self.particles)):
                r=self.particles[k]
                r.move_in_circle()
                #check collision and assume billiard physics
                r = adjust_to_collision(collision, r)
                #if r.x>=self.minX and r.x<=self.maxX and r.y<=self.maxY and r.y>=self.minY:
                new_robots.append(r)
            self.particles = new_robots

            # resample based on heading
            weights = []
            N = len(self.particles)
            for i in range(0, N):
                weights.append(heading_prob(trueh, self.particles[i].heading))
            self.particles =self.resample(weights,N)
            #print trueh, self.get_average_properties(self.particles)[2]

            #resample based on coordinates
            weights =[]
            N = len(self.particles)
            for i in range(0,N):
                weights.append(measurement_prob(new_data,[self.particles[i].x, self.particles[i].y]))
            self.particles = self.resample(weights, N)

            self.latest_data = get_average_properties(self.particles)
            train_log.append(self.latest_data)
            #print(self.get_state_difference(self.latest_data[0],self.latest_data[1], new_data[0], new_data[1])[1])
        return train_log

    def resample(self,weights,N):
        #Resample using a resampling wheel. Adopted from code from lecture.

        new_robots = []
        index = int(random.random() * N)
        beta = 0.0
        mw = max(weights)
        for i in range(0, 150):
            beta += random.random() * 2.0 * mw
            while beta > weights[index]:
                beta -= weights[index]
                index = (index + 1) % N
            r = robot(self.particles[index].x + np.random.normal(0.0, 10),
                      self.particles[index].y + np.random.normal(0.0, 10),
                      self.particles[index].heading + np.random.normal(0.0, pi / 30),
                      self.particles[index].turning+ np.random.normal(0.0, pi / 100), self.particles[index].distance)
            new_robots.append(r)
        return new_robots


    def hexbug_head(self, x, y, heading):
        #Get Hexbug head coordinates from centroid

        headx = (self.hexbug_length/2.0)*cos(heading)+x
        heady = (self.hexbug_length/2.0)*sin(heading)+y

        return headx, heady

    def centroid_from_head(self,headx,heady, heading):
        #Getting centroid coordinates from head coordinates

        x = headx - (self.hexbug_length/2.0)*cos(heading)
        y = heady - (self.hexbug_length/2.0)*sin(heading)
        return x, y

    def get_collisions(self, data):
        #Helper function for collisions

        collisions=[]
        i=2
        while i <len(data):
            curr = data[i]
            prev = data[i-1]
            h, d = get_state_difference(prev[0], prev[1], curr[0], curr[1])
            hexbug_head = self.hexbug_head(curr[0], curr[1], h)
            collision = self.detect_collision(hexbug_head)
            collisions.append([collision, h, d])
            i+=1
        #print collisions
        return collisions

    def predict_next_move(self, pos_t=0.1243, neg_t=-0.1067):
        #Make predictions based on previous particle state
        predictions = []
        count=0
        while count<60:
            new_robots = []
            new_data = self.latest_data
            #print get_average_properties(self.particles)
            hexbug_head = self.hexbug_head(self.latest_data[0], self.latest_data[1], self.latest_data[2])
            collision = self.detect_collision(hexbug_head, noise=0)
            for k in range(0, len(self.particles)):
                r = self.particles[k]
                #move according to learned turning angles, divided by correction factor.
                if r.heading>0:
                    r.move(turning=pos_t/4, distance=r.distance)
                else:
                    r.move(turning=-neg_t/4, distance=r.distance)
                # check collision and assume billiard physics
                r = adjust_to_collision([collision], r)
                new_robots.append(r)
            self.particles = new_robots
            self.latest_data = get_average_properties(self.particles)
            predictions.append([int(self.latest_data[0]),int(self.latest_data[1])])
            #print self.latest_data
            count+=1
        return predictions

    def detect_collision(self,hexbug_head, noise=4):
        #Detects if hexbug is about to collide

        if hexbug_head[0] < (self.minX + noise):
            return self.left_coll
        elif hexbug_head[0] > (self.maxX - noise):
            return self.right_coll
        elif hexbug_head[1] < (self.minY + noise):
            return self.top_coll
        elif hexbug_head[1] > (self.maxY - noise):
            return self.bottom_coll
        elif ((hexbug_head[0] - self.candle_center_x) ** 2 + (hexbug_head[1] - self.candle_center_y) ** 2) < ((self.radius) ** 2):
            return [self.candle, hexbug_head[0], hexbug_head[1]]
        else:
            return self.no_coll


if __name__ == "__main__":
    t = time.time()
    print 'learning training data'
    training = readFile('Final_Project_Resources-2016-07-16' + os.sep + 'Final_Project_Resources' + os.sep + 'training_data.txt')
    avgd, avg_pos_t, avg_neg_t = learn(training)
    total_average=0
    print avg_neg_t, avg_pos_t
    for u in range(0,50):
        total_score = []
        for k in range(1,9):
            file = readFile('Final_Project_Resources-2016-07-16' + os.sep + 'Final_Project_Resources' + os.sep + 'inputs-txt' + os.sep + 'test0'+str(k)+'.txt')
            avgerror = 0
            baseError = 0
            #print len(file),
            for i in range(1):
                x = int(floor(random.random()*1500))+100
                p = ParticleFilter(file[:x], avgd, 0)
                save((file[x:x + 60]), 'actual.txt')
                save(smooth(file[x:x + 60]), 'actual_smooth.txt')
                train_log=p.train()
                predictions = p.predict_next_move(avg_pos_t, avg_neg_t)
                save(predictions,'prediction.txt')
                actual = file[x:x+60]
                baseline = [file[x]]*60
                avgerror+=error(predictions, actual)
                baseError+=error(predictions, baseline)
            total_score.append(avgerror)
        total_score = (sum(total_score)-max(total_score)-min(total_score))/7
        total_average+=total_score
        print total_score
    print total_average/50
    print (time.time() - t)/50
