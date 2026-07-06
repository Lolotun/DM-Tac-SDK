import numpy as np
data = np.load("dataset/sample_00000.npz")
depth = data["depth"]
force = data["force"]
print("depth shape:", depth.shape)
print("depth:", depth)
print("force:", force)