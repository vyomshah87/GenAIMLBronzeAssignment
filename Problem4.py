import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
area = np.random.uniform(500, 3000, 50)
bedrooms = np.random.randint(1, 6, 50)

price = 50000 + (area * 150) + (bedrooms * 20000)

area_grid, bedroom_grid = np.meshgrid(
    np.linspace(area.min(), area.max(), 20),
    np.linspace(bedrooms.min(), bedrooms.max(), 20)
)

price_grid = 50000 + (area_grid * 150) + (bedroom_grid * 20000)

fig = plot.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(area, bedrooms, price, color='green', label='Actual Data')
ax.plot_surface(area_grid, bedroom_grid, price_grid, alpha=0.6)

ax.set_xlabel("Area")
ax.set_ylabel("No of Bedrooms")
ax.set_zlabel("House's Price")
ax.set_title("3D Regression Prediction Model for House Price")

plot.legend()
plot.show()
 