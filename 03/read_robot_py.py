import sys
sys.path.append("../scripts")
from ideal_robot import *

if __name__ == '__main__':
    world = World(10, 0.1)

    m = Map()
    m.append_landmark(Landmark(2, -2))
    m.append_landmark(Landmark(-1, -3))
    m.append_landmark(Landmark(3, 3))
    world.append(m)

    straight = Agent(0.2, 0.0)
    circling = Agent(0.2, 10.0 / 180 * math.pi)
    robot1 = IdealRobot(np.array([2, 3, math.pi / 6]).T, sensor = IdealCamera(m), agent = straight)
    robot2 = IdealRobot(np.array([-2, -1, math.pi / 5 * 6]).T, sensor = IdealCamera(m), agent = circling, color = "red")
    world.append(robot1)
    world.append(robot2)
    world.draw()