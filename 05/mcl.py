import sys
sys.path.append("../scripts")
from robot import *
from scipy.stats import multivariate_normal

class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

class Mcl:
    def __init__(self, init_pose, num, motion_noise_stds):
        self.particles = [Particle(init_pose) for i in range(num)]
        v = motion_noise_stds
        c = np.diag([v["nn"] **2, v["no"] ** 2, v["on"] ** 2, v["oo"] ** 2])
        self.motion_noise_rate_pdf = multivariate_normal(cov = c)

    def motion_update(self, nu, omega, time):
        print(self.motion_noise_rate_pdf.cov)

    def draw(self, ax, elems):
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color = "blue", alpha = 0.5))

class EstimationAgent(Agent):
    def __init__(self, time_interval, nu, omega, estimetor):
        super().__init__(nu, omega)
        self.estimator = estimator
        self.time_interval = time_interval

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

initial_pose = np.array([0, 0, 0]).T
estimator = Mcl(initial_pose, 100, motion_noise_stds = {"nn":0.01, "no":0.02, "on":0.03, "oo":0.04})
a = EstimationAgent(0.1, 0.2, 10.0 / 180 * math.pi, estimator)
estimator.motion_update(0.2, 10.0 / 180 * math.pi, 0.1)

print(np.diag([1, 2]))

'''
world = World(30, 0.1)

m = Map()
for ln in [(-4, 2), (2, -3), (3, 3)]:
    m.append_landmark(Landmark(*ln))
world.append(m)

initial_pose = np.array([2, 2, math.pi / 6]).T
estimator = Mcl(initial_pose, 100)
circling = EstimationAgent(0.2, 10.0 / 180 * math.pi, estimator)
r = Robot(initial_pose, sensor = Camera(m), agent = circling)
world.append(r)
world.draw()
'''