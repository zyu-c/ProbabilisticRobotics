import sys
sys.path.append("../scripts")
from robot import *
import copy

world = World(40.0, 0.1)
initial_pose = np.array([0, 0, 0]).T
robots = []

for i in range(100):
    r = Robot(initial_pose, sensor = None, agent = Agent(0.1, 0.0))
    world.append(r)
    robots.append(r)

world.draw()

import pandas as pd

poses = pd.DataFrame([[math.sqrt(r.pose[0] ** 2 + r.pose[1] ** 2), r.pose[2]] for r in robots], columns = ['r', 'theta'])
print(poses.transpose())

print(poses["r"].var())
print(poses["r"].mean())
print(math.sqrt(poses["r"].var() / poses["r"].mean()))