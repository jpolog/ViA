import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

#read the image
img = plt.imread('locate.jpg')


# Create a 3D grid
x, y, z = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]), np.arange(2))
# Create a figure
fig = plt.figure()
# Create a 3D subplot
#ax = fig.add_subplot(111, projection='3d')

ax = Axes3D(fig)
x = [0,1,1,0]
y = [0,0,1,1]
z = [0,0,0,0]
verts = [list(zip(x,y,z))]
ax.add_collection3d(Poly3DCollection(verts))

plt.show()
