import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as TF
from utils_patch import PatchApplier_v1
from ptop import ParticleToPatch_One


class OptimizeFunction:
    def __init__(self, detector, patch_size, device):
        self.detector = detector
        self.device = device
        self.ptp = ParticleToPatch_One(patch_size)
        self.pa = PatchApplier_v1()
        self.size = 0
        self.num_patch = 1
        self.patch_size = patch_size
        
    def set_para(self, targets, imgs):
        self.targets = targets
        self.imgs = imgs
        
    def evaluate(self, x):
        # x: (num, dim)
        with torch.no_grad():
            patch_tf, patch_mask_tf = self.ptp(x, self.targets, self.imgs)
            imgWithPatch = self.pa(self.imgs, patch_tf, patch_mask_tf)
            
            out, train_out = self.detector(imgWithPatch)
            obj_confidence = out[:, :, 4]
            max_obj_confidence, _ = torch.max(obj_confidence, dim=1)
            obj_loss = torch.mean(max_obj_confidence)
            
            num_block = torch.sum(x[0])
            current_size = self.num_patch * self.patch_size * self.patch_size * num_block / 9
            delta_size = current_size - self.size
            if delta_size > 0:
                return_obj_loss = obj_loss + delta_size * 0.1
            else:
                return_obj_loss = obj_loss
            self.size = current_size

        return return_obj_loss
        

class SwarmParameters:
    pass


class Particle:
    def __init__(self, dimensions, device):
        self.device = device
        self.dimensions = dimensions
        self.w = 0.5
        self.c1 = 2
        self.c2 = 2
        classes = 2
        
        random_matrix = torch.rand((3, 3)).to(self.device)
        random_matrix[random_matrix>=0.5] = 1
        random_matrix[random_matrix<0.5] = 0
        state_matrix = random_matrix
        
        position = torch.rand(dimensions, classes).to(self.device)
        
        self.position = [state_matrix, position]
        self.velocity = torch.zeros((dimensions, classes)).to(self.device)
        
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")]).to(self.device)
        
    
    def update_velocity(self, gbest_position):
        r1 = torch.rand(1).to(self.device)
        r2 = torch.rand(1).to(self.device)
        for i in range(0, self.dimensions):
            self.velocity[i] = self.w * self.velocity[i] \
                               + self.c1 * r1 * (self.pbest_position[1][i] - self.position[1][i]) \
                               + self.c2 * r2 * (gbest_position[1][i] - self.position[1][i])

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters
        
        
    def move(self):
        for i in range(0, self.dimensions):
            self.position[1][i] = self.position[1][i] + self.velocity[i]
            
        random_matrix = torch.rand((3, 3)).to(self.device)
        random_matrix[random_matrix>=0.5] = 1
        random_matrix[random_matrix<0.5] = 0
        self.position[0] = random_matrix
        self.position[1].data.clamp_(0,1)
        

class PSO:
    def __init__(self, swarm_size, device):
        self.max_iterations = 3
        self.swarm_size = swarm_size

        self.gbest_position = [0, 0]
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        self.swarm = []
        for i in range(self.swarm_size):
            self.swarm.append(Particle(dimensions=1, device=device))         # dimension
        
    
    def optimize(self, function):
        self.fitness_function = function
        
        
    def run(self):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0
        # --- Run
        for iteration in range(self.max_iterations):
            # --- Set PBest
            for particle in self.swarm:
                fitness_candidate = self.fitness_function.evaluate(particle.position)
                #break
                if (particle.pbest_value > fitness_candidate):
                    particle.pbest_value = fitness_candidate
                    particle.pbest_position[0] = particle.position[0].clone()
                    particle.pbest_position[1] = particle.position[1].clone()
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_candidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_candidate:
                    self.gbest_value = best_fitness_candidate
                    self.gbest_position[0] = particle.position[0].clone()
                    self.gbest_position[1] = particle.position[1].clone()
                    self.gbest_particle = copy.deepcopy(particle)
            
            r1s = []
            r2s = []
            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()

        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        swarm_parameters.c1 = self.gbest_particle.c1
        swarm_parameters.c2 = self.gbest_particle.c2
        return swarm_parameters
        