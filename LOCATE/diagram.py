import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = plt.imread('locate.jpg')

# Create a 3D grid
x, y, z = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]), np.arange(2))

# Create a figure
fig = plt.figure()

# Create a 3D subplot
ax = fig.add_subplot(111, projection='3d')

# Plot the image on the grid
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=img.reshape(-1, 4)/255.0, marker='.', s=2)

# Set the axis limits
ax.set_xlim([0, img.shape[1]])
ax.set_ylim([0, img.shape[0]])
ax.set_zlim([0, 2])

# Hide the ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Show the plot
plt.show()

