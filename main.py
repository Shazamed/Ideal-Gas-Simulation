import random
import matplotlib.pyplot as plt
import numpy as np
import pygame
import math
from matplotlib.animation import FuncAnimation
import sys
from PIL import Image
import math

dt = 0.5
SIZE = BOX_WIDTH, BOX_HEIGHT = 500, 500
acceleration = 0

def image_colour(coordinates):
    marisad = Image.open("marisad.jpg")
    colour = marisad.getpixel(((coordinates[0]-250)/200 * 240, (coordinates[1] - 250)/200 * 240))
    return colour



def bubble_sort(particle_list, axis, start):
    for i in range(len(particle_list)):
        swapped = False
        for j in range(len(particle_list) - i - 1):
            if particle_list[j].p[axis] - start * particle_list[j].r > particle_list[j+1].p[axis] - start * particle_list[j+1].r:
                temp = particle_list[j]
                particle_list[j] = particle_list[j+1]
                particle_list[j+1] = temp
                swapped = True
        if not swapped:
            break


def particle_gen(n):
    particles = []
    # particles.append(Particle(np.array([250., 250.]), np.array([0., 0.]), 30, 10000, (0, 0, 0)))
    square_side = math.ceil(n**0.5)
    for i in range(n):
        coordinates = np.array([random.random()*200, 300+random.random()*200])
        # print([(i%square_side/square_side) * 200, (i/square_side**2) * 200])
        # coordinates = np.array([(i%square_side/square_side) * 200 + 10, (i/square_side**2) * 200 + 10])
        particles.append(Particle(coordinates, np.array([1., 1.]), 2, 1, (0,0,0)))# np.random.uniform(-1,1,2)
    return particles

def marisad(n):
    particles = []
    particles.append(Particle(np.array([30., 30.]), np.array([1., 1.]), 30, 10000, (0, 0, 0)))
    square_side = math.ceil(n**0.5)
    for i in range(n):
        coordinates = np.array([(i%square_side/square_side) * 200 + 250, (math.floor(i/square_side)/square_side) * 200 + 250])
        particles.append(Particle(coordinates, np.array([0., 0.]), 2, 1, image_colour(coordinates))) # np.random.uniform(-1,1,2)
    return particles

def particle_gen2(n):
    particles = []
    particles.append(Particle(np.array([200., 200.]), np.array([0., 0.]), 30, 10000, (0, 0, 0)))
    for i in range(n):
        coordinates = np.array([200 + math.cos(math.radians(360 * i / n))*100, 200 + math.sin(math.radians(360 * i/n))*100])
        velocity = np.array([-math.cos(math.radians(360 * i / n))*1, -math.sin(math.radians(360 * i/n))*1])
        particles.append(Particle(coordinates, velocity, 3, 1, (0,0,0))) #np.random.rand(3)*255))
    return particles




class Particle:
    def __init__(self, pos, vel, rad, mass, colour):
        self.p = pos
        self.v = vel
        self.r = rad
        self.m = mass
        self.colour = colour

    def velocity_update(self):
        self.p += self.v * dt

    def acceleration(self):
        self.v[1] += acceleration * dt

    def collision_wall_check(self):
        if self.p[0] + self.r > BOX_WIDTH or self.p[0] - self.r < 0:
            self.v[0] *= -1
        if self.p[1] + self.r > BOX_HEIGHT or self.p[1] - self.r < 0:
            self.v[1] *= -1

    def collision_particle_check(self, col_particle):
        p_diff = self.p - col_particle.p
        if np.linalg.norm(p_diff) < self.r + col_particle.r:
            v_diff = self.v - col_particle.v
            dv = p_diff * np.dot(v_diff, p_diff) / (np.linalg.norm(p_diff)**2)
            original_v1 = self.v
            original_v2 = col_particle.v
            self.v -= 2 * col_particle.m / (col_particle.m + self.m) * dv
            col_particle.v += 2 * self.m / (col_particle.m + self.m) * dv
            # if np.linalg.norm(p_diff + (self.v - col_particle.v) * dt) > np.linalg.norm(p_diff):
            #     print(np.linalg.norm(p_diff + (self.v - col_particle.v) * dt), np.linalg.norm(p_diff))
            #     self.v = original_v1
            #     col_particle.v = original_v2


        # if np.linalg.norm(self.p-col_particle.p) < self.r + col_particle.r:
        #     original_v = self.v
        #     self.v = ((self.m-col_particle.m)/(self.m+col_particle.m)) * self.v + (2*col_particle.m/(self.m+col_particle.m)) * col_particle.v
        #     col_particle.v = ((col_particle.m - self.m) / (self.m + col_particle.m)) * col_particle.v + (
        #                 2 * self.m / (self.m + col_particle.m)) * original_v



def sweep_and_prune(x_particle_list):
    active = []
    for i in range(len(x_particle_list)):
        if not active:
            active.append(x_particle_list[i])
        else:
            active_list = active.copy()
            for j in active_list:
                if (j.p[0] + j.r > x_particle_list[i].p[0]-x_particle_list[i].r):
                    j.collision_particle_check(x_particle_list[i])

                    pass
                else:
                    active.pop(0)
            active.append(x_particle_list[i])

            bubble_sort(active, 0, -1)
        # if len(active) > 1:
        #     print(len(active))



if __name__ == '__main__':
    screen = pygame.display.set_mode(SIZE)
    yinyang = pygame.image.load("Yin_and_Yang_symbol.png")
    yinyang = pygame.transform.scale(yinyang, (60, 60))
    # particle_list = marisad(700)
    particle_list = particle_gen(500)
    # fig = plt.figure()

    start = False
    angle = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        screen.fill((255,255,255))

        for particle in particle_list:
            particle.acceleration()
            particle.velocity_update()
            particle.collision_wall_check()
            # plt.scatter(particle.p[0], particle.p[1])
            pygame.draw.circle(screen, particle.colour, particle.p, particle.r)
            if particle.m == 10000:
                yinyang_coord = particle.p-30

        # rotated_yinyang = pygame.transform.rotate(yinyang, angle)
        # new_rect = rotated_yinyang.get_rect(center=yinyang.get_rect(topleft=yinyang_coord).center)
        # angle -= 10
        # screen.blit(rotated_yinyang,  new_rect)

        bubble_sort(particle_list, axis=0, start=1)
        sweep_and_prune(particle_list)

        # plt.ylim(ymax=100, ymin=0)
        # plt.xlim(xmax=100, xmin=0)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.clf()
        # plt.close(fig)
        pygame.display.flip()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
