import random

samples = [random.choice([1, 2, 3, 4, 5, 6]) for i in range(1000)]
print(sum(samples) / len(samples))