import igl
import scipy as sp
import numpy as np
from meshplot import plot, subplot

import os
root_folder = os.getcwd()

# ----------------------------------------------------------------------------------------------------

v, f  = igl.read_triangle_mesh(os.path.join(root_folder, "head.obj"))
## Find the open boundary
bnd = igl.boundary_loop(f)



## Map the boundary to a circle, preserving edge proportions
bnd_uv = igl.map_vertices_to_circle(v, bnd)

## Harmonic parametrization for the internal vertices
uv = igl.harmonic_weights(v, f, bnd, bnd_uv, 1)
v_p = np.hstack([uv, np.zeros((uv.shape[0],1))])

p = subplot(v, f, uv=uv, shading={"wireframe": False, "flat": False}, s=[1, 2, 0])
subplot(v_p, f, uv=uv, shading={"wireframe": True, "flat": False}, s=[1, 2, 1], data=p)

p.save("Harmonic.html")

# @interact(mode=['3D','2D'])
# def switch(mode):
#     if mode == "3D":
#         plot(v, f, uv=uv, shading={"wireframe": False, "flat": False}, plot=p)
#     if mode == "2D":
#         plot(v_p, f, uv=uv, shading={"wireframe": True, "flat": False}, plot=p)

# ----------------------------------------------------------------------------------------------------

v, f = igl.read_triangle_mesh(os.path.join(root_folder, "head.obj"))

# Fix two points on the boundary
b = np.array([2, 1])

bnd = igl.boundary_loop(f)
b[0] = bnd[0]
b[1] = bnd[int(bnd.size / 2)]

bc = np.array([[0.0, 0.0], [1.0, 0.0]])

# LSCM parametrization
_, uv = igl.lscm(v, f, b, bc)

p = subplot(v, f, uv=uv, shading={"wireframe": False, "flat": False}, s=[1, 2, 0])
subplot(uv, f, uv=uv, shading={"wireframe": False, "flat": False}, s=[1, 2, 1], data=p)
p.save("LSCM.html")

# @interact(mode=['3D','2D'])
# def switch(mode):
#     if mode == "3D":
#         plot(v, f, uv=uv, shading={"wireframe": False, "flat": False}, plot=p)
#     if mode == "2D":
#         plot(uv, f, uv=uv, shading={"wireframe": True, "flat": False}, plot=p)

# ----------------------------------------------------------------------------------------------------

v, f  = igl.read_triangle_mesh(os.path.join(root_folder, "head.obj"))

## Find the open boundary
bnd = igl.boundary_loop(f)

## Map the boundary to a circle, preserving edge proportions
bnd_uv = igl.map_vertices_to_circle(v, bnd)

## Harmonic parametrization for the internal vertices
uv = igl.harmonic_weights(v, f, bnd, bnd_uv, 1)

arap = igl.ARAP(v, f, 2, np.zeros(0))
uva = arap.solve(np.zeros((0, 0)), uv)

p = subplot(v, f, uv=uva, shading={"wireframe": False, "flat": False}, s=[1, 2, 0])
subplot(uva, f, uv=uva, shading={"wireframe": False, "flat": False}, s=[1, 2, 1], data=p)

p.save("As_grid_as_possible.html")

# @interact(mode=['3D','2D'])
# def switch(mode):
#     if mode == "3D":
#         plot(v, f, uv=uva, shading={"wireframe": False, "flat": False}, plot=p)
#     if mode == "2D":
#         plot(uva, f, uv=uva, shading={"wireframe": True, "flat": False}, plot=p)