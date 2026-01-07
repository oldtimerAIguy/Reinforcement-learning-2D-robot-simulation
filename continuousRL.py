# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 09:05:24 2019

@author: kmcfall
"""

import math
import pickle
import random
import copy
import numpy as np
import matplotlib.pyplot as plot
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, stateLen, actionLen):
        super(DQN, self).__init__()
        nodes = 100
        self.h1 = nn.Linear(stateLen, nodes)
        self.h2 = nn.Linear(nodes, nodes)
        self.output = nn.Linear(nodes, actionLen)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = self.output(x)
        #return F.log_softmax(x, dim=1)
        return x



def select_action(state):
    global steps_done
    sample = random.random()
    c = 50*counter[getGridPos(x,y,pose)]
    if len(c) == 0:
        c = 0
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done+c) / EPS_DECAY)
    #eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * (steps_done) / EPS_DECAY)
    #m = np.max(counter)
    #eps_threshold = 1-c/m
    steps_done += 1
    if draw and i_episode % drawFreq == 0:
        print(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def updatePose(pose, steer):
    L = 2 # (m) Distance between front and rear axles
    R = L/np.arctan(steer)
    ds = stepSize # (m)
    dq = ds/R
    pose = np.array((pose[0] + ds*np.cos(pose[2]+dq/2),
                     pose[1] + ds*np.sin(pose[2]+dq/2),
                     pose[2] + dq))
    if pose[2] > np.pi:
        pose[2] -= 2*np.pi
    if pose[2] < -1*np.pi:
        pose[2] += 2*np.pi
    return pose

def dist(p1,p2):
    return np.sqrt(sum((p1[0:2]-p2[0:2])**2))
  
def processAction(action, pose):
    if action == 0:
        newPose = updatePose(pose,  40*np.pi/180)
    if action == 1:
        newPose = updatePose(pose,   0.00001    )
    if action == 2:
        newPose = updatePose(pose, -40*np.pi/180)
    if newPose[2] > np.pi:
        newPose[2] -= np.pi
    if newPose[2] < -np.pi:
        newPose[2] += np.pi
    state = getState(pose)
    reward = 0 #10 - dist(goal,newPose[0:2])
    #print('Base',reward)
    if np.abs(getHeadDifference(newPose)) < np.abs(getHeadDifference(pose)):
        reward += 2
        #print('Heading bonus')
    if min(state[0,0:3]) < 2:
        d = min(state[0,0:3])
        val = 2*(2 - d)
        #print('Close penalty', val)
        reward -= val
    if newPose[0] < 0 or newPose[1] < 0 or newPose[0] > xMax or newPose[1] > yMax:
        newPose[0:2] = pose[0:2]
        return torch.tensor(state,dtype=torch.float).view(1,n_state), torch.tensor([-20-dist(pose[0:2],goal)],dtype=torch.float), newPose # Ran into boundary
    if hitWall(pose, newPose):
        newPose[0:2] = pose[0:2]
        return torch.tensor(state,dtype=torch.float).view(1,n_state), torch.tensor([-20-dist(pose[0:2],goal)],dtype=torch.float), newPose # Ran into boundary
    if dist(newPose, goal) < dist(pose, goal):
        reward +=  2
    if atGoal(newPose,goal):
        reward = 20
    return  torch.tensor(state,dtype=torch.float).view(1,n_state), torch.tensor([reward],dtype=torch.float), newPose

def getHeadDifference(pose):
    goalHead = np.arctan2(goal[1]-pose[1],goal[0]-pose[0])
    headDiff = goalHead - pose[2]
    if headDiff > np.pi:
        headDiff -= np.pi
    if headDiff < -np.pi:
        headDiff += np.pi
    return headDiff

def getState(pose):
    headDiff = getHeadDifference(pose)
    closestAhead = 500
    closestLeft  = 500
    closestRight = 500
    for i in range(boundaries.shape[0]):
        poseLeft = copy.copy(pose)
        poseLeft[2] += 20*np.pi/180
        cross, xI, yI = lineCross(poseLeft,boundaries[i,:])
        d = dist(pose[0:2],(xI,yI))
        if cross and d<closestLeft:
            closestLeft = d
            xLeft = xI
            yLeft = yI
        cross, xI, yI = lineCross(pose,boundaries[i,:])
        d = dist(pose[0:2],(xI,yI))
        if cross and d<closestAhead:
            closestAhead = d
            xAhead = xI
            yAhead = yI
        poseLeft[2] -= 40*np.pi/180
        cross, xI, yI = lineCross(poseLeft,boundaries[i,:])
        d = dist(pose[0:2],(xI,yI))
        if cross and d<closestRight:
            closestRight = d
            xRight = xI
            yRight = yI
    state = [closestLeft,closestAhead,closestRight,headDiff]
    #if draw:
    #    plot.plot((pose[0],xLeft ),(pose[1],yLeft ),'r-')
    #    plot.plot((pose[0],xRight),(pose[1],yRight),'r-')
    #    plot.plot((pose[0],xAhead),(pose[1],yAhead),'r-')
    return torch.tensor(state,dtype=torch.float).view(1,n_state)

def atGoal(poseNow, poseGoal):
    return np.sqrt(sum((poseNow[0:2]-poseGoal[0:2])**2)) < stepSize

def drawArrow(pose,size=0.5,color='b',zorder = 10):
    dx = size*np.cos(pose[2])
    dy = size*np.sin(pose[2])
    plot.arrow(pose[0],pose[1],dx,dy, color = color, width = size*0.15, zorder = zorder)

def GTE(x,y):
    if x>y or np.abs(x-y)<1e-6:
        return True
    else:
        return False

def LTE(x,y):
    if x<y or np.abs(x-y)<1e-6:
        return True
    else:
        return False

def lineCross(pose, points):
    x1 = points[0]
    y1 = points[1]
    x2 = points[2]
    y2 = points[3]
    x4 = pose[0]+np.cos(pose[2])
    y4 = pose[1]+np.sin(pose[2])
    xI,yI = getIntersection(x1,y1, x2,y2, pose[0],pose[1], x4,y4)
    if ((GTE(x4,pose[0]) and GTE(xI,pose[0])) or (LTE(x4,pose[0]) and LTE(xI,pose[0]))) and GTE(xI,min(x1,x2)) and LTE(xI,max(x1,x2)) and GTE(yI,min(y1,y2)) and LTE(yI,max(y1,y2)):
        return True, xI, yI
    else:
        return False, xI, yI

def getIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    return ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom , ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom

def hitWall(pose, newPose):
    for i in range(boundaries.shape[0]):
        x1 = boundaries[i,0]
        y1 = boundaries[i,1]
        x2 = boundaries[i,2]
        y2 = boundaries[i,3]
        x4 = pose[0]+np.cos(pose[2])
        y4 = pose[1]+np.sin(pose[2])
        xI,yI = getIntersection(x1,y1, x2,y2, pose[0],pose[1], x4,y4)
        if x2 == x1:
            m = 99e99
        else:
            m = (y2-y1)/(x2-x1)
        b = y1 - m*x1
        if pose[1] > m*pose[0]+b and newPose[1] < m*newPose[0]+b:
            if ((GTE(x4,pose[0]) and GTE(xI,pose[0])) or (LTE(x4,pose[0]) and LTE(xI,pose[0]))) and GTE(xI,min(x1,x2)) and LTE(xI,max(x1,x2)) and GTE(yI,min(y1,y2)) and LTE(yI,max(y1,y2)):
                return True
        if pose[1] < m*pose[0]+b and newPose[1] > m*newPose[0]+b:
            if ((GTE(x4,pose[0]) and GTE(xI,pose[0])) or (LTE(x4,pose[0]) and LTE(xI,pose[0]))) and GTE(xI,min(x1,x2)) and LTE(xI,max(x1,x2)) and GTE(yI,min(y1,y2)) and LTE(yI,max(y1,y2)):
                return True
    return False

def drawBoundaries(boundaries):
    plot.plot(goal[0],goal[1],'*')
    for i in range(boundaries.shape[0]):
        plot.plot([boundaries[i,0],boundaries[i,2]],[boundaries[i,1],boundaries[i,3]],'k-')

def getGridPos(x,y,pose):
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]
    return np.logical_and(pose[0] > x-dx/2 , np.logical_and(pose[0] < x+dx/2 , np.logical_and(pose[1] > y-dy/2, pose[1] < y+dy/2)))

def coordTransform(x1,y1 , x2,y2 , x,y):
    nx = x.shape[0]
    ny = x.shape[1]
    mx = (nx-1)/(np.max(x)-np.min(x))
    bx = -1*mx*np.min(x)
    my = (ny-1)/(np.min(y)-np.max(y))
    by = -1*my*np.max(y)
    return mx*x1+bx , my*y1+by , mx*x2+bx , my*y2+by

def visualize(counter, x,y, policy):
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]
    im = copy.deepcopy(counter)
    im[im>np.mean(im)] = np.mean(im)
    plot.imshow(np.flip(im/np.max(im),0),cmap='gray')
    ang = np.linspace(0,2*np.pi,9)
    ang = ang[0:-1]
    color = 'cbm'
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):            
            for theta in ang:
                pose = np.array([x[row,col], y[row,col], theta])
                state = getState(pose)
                action = policy_net(state).max(1)[1].view(1,1)
                x1,y1,x2,y2 = coordTransform(x[row,col]                   , y[row,col],
                                             x[row,col]+dx/2*np.cos(theta), y[row,col]+dy/2*np.sin(theta), x,y)
                plot.plot((x1,x2), (y1,y2), color[action]+'-')
    plot.pause(0.001)


stepSize = 0.5
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 170000
TARGET_UPDATE = 100
n_actions = 3
n_state = 4
memory = ReplayMemory(10000)
xMax = 10
yMax = 10
#boundaries = np.array([[0,0,xMax,0],[xMax,0,xMax,yMax],[0,yMax,xMax,yMax],[0,0,0,yMax],[6,4,6,7],[6,7,9,7],[9,7,9,4],[9,4,6,4]])
#boundaries = np.array([[0,0,xMax,0],[xMax,0,xMax,yMax],[0,yMax,xMax,yMax],[0,0,0,yMax],[4,4,4,7],[4,4,7,4]])
boundaries = np.array([[0,0,xMax,0],[xMax,0,xMax,yMax],[0,yMax,xMax,yMax],[0,0,0,yMax],[4,4,4,7],[4,4,7,4]])
#boundaries = np.array([[0,0,xMax,0],[xMax,0,xMax,yMax],[0,yMax,xMax,yMax],[0,0,0,yMax]])
num_episodes = 200000
goal = np.array([9,9])
draw = True
drawFreq = 50
gridSize = 19
x,y = np.meshgrid(np.linspace(0,xMax,gridSize+2),np.linspace(0,yMax,gridSize+2))
x = x[1:-1,1:-1]
y = y[1:-1,1:-1]
best = 100

#while sum(log) > 15 and EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) > 1.5*EPS_END: # and i_episode < num_episodes
folder = 'both'
file = open(  folder+"/boundSave.p","wb")
pickle.dump(boundaries,file)
file.close()
results = []
steps = []
for count in range(4,10):
    i_episode = 0
    policy_net = DQN(n_state, n_actions)
    target_net = DQN(n_state, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters())
    counter = np.zeros_like(x)
    steps_done = 0
    plot.figure(2)
    plot.clf()
    tick = 0
    log = 100*[1]
    results.append([])
    while sum(log)>15 and i_episode<150000: #EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY) > 1.5*EPS_END: # and i_episode < num_episodes
        #pose = np.array([1+(0.5-np.random.random())*2,1+(0.5-np.random.random())*2,np.pi/4+(0.5-np.random.random())*2])
        tick = max(tick-1,0)
        pose = np.array([1,1,np.pi/4+(0.5-np.random.random())/5])
        #pose = np.array([1,1,np.pi/4])
        #pose = np.array([1+np.random.random()*5,1+np.random.random()*5,np.pi/4+(0.5-np.random.random())])
        state = getState(pose)
        if draw and i_episode % drawFreq == 0:
            plot.figure(1)
            plot.clf()
            drawBoundaries(boundaries)
        reward = [0]
        t = 0
        while np.abs(reward[0])<19.9 and t < 100:
            t += 1
            counter[getGridPos(x,y,pose)] += 1
            if draw and i_episode % drawFreq == 0:
                plot.figure(1)
                drawArrow(pose)
                plot.axis((-1,xMax+1,-1,yMax+1))
            action = select_action(state)
            next_state, reward, pose = processAction(action, pose)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            # Move to the next state
            state = next_state
            if draw and i_episode % drawFreq == 0:
                drawArrow(pose,color='r')
                #print(reward.detach().numpy()[0])
                #print(' ')
                plot.axis('equal')
                plot.pause(0.001)       
            # Perform one step of the optimization (on the target network)
            optimize_model()
        thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        d = dist(pose[0:2],goal)
        results[len(results)-1].append(np.sum(log))
        log = log[1:]
        if reward>19.9:
            log.append(0)
            print('GOAL!', np.sum(log), t, np.round(thresh,2))
        elif reward[0]<-19.9:
            log.append(1)
            print('Hit wall', np.sum(log), np.round(d,2), np.round(thresh,2))                
        else:
            log.append(1)
            print('Max iter fail', np.sum(log), np.round(d,2), np.round(thresh,2))
        if i_episode % TARGET_UPDATE == 0:
            print("update")
            target_net.load_state_dict(policy_net.state_dict())
        if np.sum(log) < best and tick == 0:
            print('Save',np.sum(log))
            #if draw:
            #    plot.figure(2)
            #    plot.clf()
            #    visualize(counter,x,y,policy_net)
            #    plot.pause(0.001)
            tick = 50
            file = open( folder+"/policySave"+str(count)+".p","wb")
            pickle.dump(policy_net,file)
            file.close()
            file = open(folder+"/counterSave"+str(count)+".p","wb")
            pickle.dump(counter   ,file)
            file.close()
        best = min(best,np.sum(log))
        i_episode += 1
    print('Complete',count)
    steps.append(steps_done)
    file = open( folder+"/steps.p","wb")
    pickle.dump(steps,file)
    file.close()
    file = open( folder+"/results.p","wb")
    pickle.dump(results,file)
    file.close()
    file = open( folder+"/policySave"+str(count)+".p","wb")
    pickle.dump(policy_net,file)
    file.close()
    file = open(folder+"/counterSave"+str(count)+".p","wb")
    pickle.dump(counter   ,file)
    file.close()
plot.ioff()
plot.show()

