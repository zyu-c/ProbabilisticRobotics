import matplotlib.pyplot as plt

class World:
    def __init__(self):
        self.objects = []
    
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        fig = plt.figure(figsize = (8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("X", fontsize = 20)
        ax.set_ylabel("Y", fontsize = 20)
        for obj in self.objects: obj.draw(ax)
        plt.show()

world = World()
world.draw()