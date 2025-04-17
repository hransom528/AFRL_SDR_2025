# Radar Visualization
# Dynamically plots a point in polar coords

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

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

# Buttons to switch between different algorithms
class Switcher():
	# Algortihm selector
	algorithms = ["triangulation", "tdoa", "ml_nobaseband", "ml_baseband"]
	alg = 0

	# TODO: Implement different data feeds from other algorithms
	# Triangulation button callback
	def triangulation(self, event):
		alg = 0

	# TDoA button callback
	def tdoa(self, event):
		alg = 1

	# ML (No Baseband) button callback
	def ml_nobaseband(self, event):
		alg = 2

	# ML w/Baseband button callback
	def ml_baseband(self, event):
		alg = 3

# MAIN
if __name__ == "__main__":
	# Creates the figure and axes object
	fig = plt.figure()
	ax = fig.add_subplot(projection='polar')

	# Creates the initial coordinate point
	r = 2
	theta = np.deg2rad(60)
	#staticRadarPlot((r, theta))

	# Adds buttons
	ax_triangulation = fig.add_axes([0.55, 0.01, 0.1, 0.075])
	ax_tdoa 		 = fig.add_axes([0.66, 0.01, 0.1, 0.075])
	ax_nobaseband 	 = fig.add_axes([0.77, 0.01, 0.1, 0.075])
	ax_baseband 	 = fig.add_axes([0.88, 0.01, 0.1, 0.075])

	callback = Switcher()
	btn_triangulation = Button(ax_triangulation, 'Triangulation')
	btn_tdoa = Button(ax_tdoa, "TDoA")
	btn_nobaseband = Button(ax_nobaseband, "ML (No Baseband)")
	btn_baseband = Button(ax_baseband, "ML w/Baseband")

	btn_triangulation.on_clicked(callback.triangulation)
	btn_tdoa.on_clicked(callback.tdoa)
	btn_nobaseband.on_clicked(callback.ml_nobaseband)
	btn_baseband.on_clicked(callback.ml_baseband)

	# Creates animation object
	anim = FuncAnimation(fig, radarUpdate)
	plt.show()