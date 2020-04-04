import matplotlib
#matplotlib.use('nbagg')
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

class World:
    def __init__(self, debug = False):
        self.objects = []
        self.debug = debug
    
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize = 10)
        ax.set_ylabel("Y", fontsize = 10)
        elems = []
        if self.debug:
            for i in range(1000):
                self.one_step(i, elems, ax)
        else:
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs = (elems, ax), frames = 10, interval = 1000, repeat = False)
            plt.show()

    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        elems.append(ax.text(-4.4, 4.5, "t = " + str(i), fontsize = 10))
        for obj in self.objects:
            obj.draw(ax, elems)

class IdealRobot:
    def __init__(self, pose, color = "black"):
        self.pose = pose
        self.r = 0.2
        self.color = color

    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x, xn], [y, yn], color = self.color)
        c = patches.Circle(xy = (x,y), radius = self.r, fill = False, color = self.color)
        elems.append(ax.add_patch(c))

    @classmethod
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array([nu * math.cos(t0), nu * math.sin(t0), omega]) * time
        else:
            return pose + np.array([nu / omega * (math.sin(t0 + omega * time) - math.sin(t0)),
                                    nu / omega * (-math.cos(t0 + omega * time) + math.cos(t0)),
                                    omega * time])

print(IdealRobot.state_transition(0.1, 0.0, 1.0, np.array([0, 0, 0]).T))
print(IdealRobot.state_transition(0.1, 10 / 180 * math.pi, 9.0, np.array([0, 0, 0]).T))
print(IdealRobot.state_transition(0.1, 10 / 180 * math.pi, 18.0, np.array([0, 0, 0]).T))