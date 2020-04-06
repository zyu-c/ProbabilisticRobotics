import sys
sys.path.append("../scripts")
from ideal_robot import *
from scipy.stats import expon, norm, uniform

class Robot(IdealRobot):
    def __init__(self, pose, agent = None, sensor = "None", color = "black", \
            noise_per_mater = 5, noise_std = math.pi / 60, \
            bias_rate_stds = (0.1, 0.1), \
            expected_stuck_time = 1e100, expected_escape_time = 1e-100, \
            expected_kidnap_time = 1e100, kindnap_range_x = (-5.0, 5.0), kindnap_range_y = (-5.0, 5.0)):
        super().__init__(pose, agent, sensor, color)
        self.noise_pdf = expon(scale = 1.0 / (1e-100 + noise_per_mater))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale = noise_std)
        self.bias_rate_nu = norm.rvs(loc = 1.0, scale = bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc = 1.0, scale = bias_rate_stds[1])
        self.stuck_pdf = expon(scale = expected_stuck_time)
        self.escape_pdf = expon(scale = expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        self.is_stuck = False
        self.kindnap_pdf = expon(scale = expected_kidnap_time)
        self.time_until_kindnap = self.kindnap_pdf.rvs()
        rx, ry = kindnap_range_x, kindnap_range_y
        self.kindnap_dist = uniform(loc = (rx[0], ry[0], 0.0), scale = (rx[1] - rx[0], ry[1] - ry[0], 2 * math.pi))

    def noise(self, pose, nu, omega, time_interval):
        self.distance_until_noise -= abs(nu) * time_interval + self.r * abs(omega) * time_interval
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
        return pose

    def bias(self, nu, omega):
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega

    def stuck(self, nu, omega, time_interval):
        if self.is_stuck:
            self.time_until_escape -= time_interval
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True
        return nu * (not self.is_stuck), omega * (not self.is_stuck)

    def kindnap(self, pose, time_interval):
        self.time_until_kindnap -= time_interval
        if self.time_until_kindnap <= 0.0:
            self.time_until_kindnap += self.kindnap_pdf.rvs()
            return np.array(self.kindnap_dist.rvs()).T
        else:
            return pose

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kindnap(self.pose, time_interval)


world = World(30, 0.1)

circling = Agent(0.2, 10.0 / 180 * math.pi)
for i in range(10):
    r = Robot(np.array([0, 0, 0]).T, sensor = None, agent = circling, color = "gray", noise_per_mater = 0, bias_rate_stds = (0.0, 0.0), expected_kidnap_time = 5)
    world.append(r)

r = IdealRobot(np.array([0, 0, 0]).T, sensor = None, agent = circling, color = "red")
world.append(r)
world.draw()