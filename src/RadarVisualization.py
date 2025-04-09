# Radar Visualization
# Dynamically plots a point in polar coords

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Convert polar coords to rectangular coords
def rect2polar(coord, degrees=True):
	# Get coord vals
	x = coord[0]
	y = coord[1]

	# Calculate polar coords
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan2(y, x)

	# Convert to degrees if desired
	if (degrees):
		theta = np.rad2deg(theta)
	
	return (r, theta)

# Convert rectangular coords to polar coords
def polar2rect(coord, degrees=True):
	r = coord[0]
	theta = coord[1]

	# Convert theta to radians if in degrees
	if (degrees):
		theta = np.deg2rad(theta)
	
	# Convert to rectangular coords
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	return (x, y)

# Plot radar coords in polar format (static)
def staticRadarPlot(coord):
	fig = plt.figure()
	ax = fig.add_subplot(projection='polar')

	r = coord[0]
	theta = coord[1]
	c = ax.scatter(theta, r)
	plt.show()

# Dynamic radar animation update function
def radarUpdate(frame):
	global theta
	theta += np.deg2rad(5)

	ax.clear()
	ax.scatter(theta, r)
	fig.canvas.draw()

# MAIN
#if __name__ == "__main__":
# Creates the figure and axes object
fig = plt.figure()
ax = fig.add_subplot(projection='polar')

# Creates the initial coordinate point
r = 2
theta = np.deg2rad(60)
#staticRadarPlot((r, theta))

# Creates animation object
anim = FuncAnimation(fig, radarUpdate)
plt.show()