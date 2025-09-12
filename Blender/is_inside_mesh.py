import numpy



'''
Naive and straightforward implementation of the inside/outside point mesh test
https://github.com/marmakoide/inside-3d-mesh
'''

def is_inside_naive(triangles, X):
	# Compute triangle vertices and their norms relative to X
	M = triangles - X
	M_norm = numpy.sqrt(numpy.sum(M ** 2, axis = 2))

	# Accumulate generalized winding number per triangle
	winding_number = 0.
	for (A, B, C), (a, b, c) in zip(M, M_norm):
		winding_number += numpy.arctan2(numpy.linalg.det(numpy.array([A, B, C])), (a * b * c) + c * numpy.dot(A, B) + a * numpy.dot(B, C) + b * numpy.dot(C, A))

	# Job done
	return winding_number >= 2. * numpy.pi



'''
Optimized for numpy implementation of the inside/outside point mesh test
'''

def is_inside_turbo(triangles, X):
	# Compute euclidean norm along axis 1
	def anorm2(X):
		return numpy.sqrt(numpy.sum(X ** 2, axis = 1))



	# Compute 3x3 determinant along axis 1
	def adet(X, Y, Z):
		ret  = numpy.multiply(numpy.multiply(X[:,0], Y[:,1]), Z[:,2])
		ret += numpy.multiply(numpy.multiply(Y[:,0], Z[:,1]), X[:,2])
		ret += numpy.multiply(numpy.multiply(Z[:,0], X[:,1]), Y[:,2])
		ret -= numpy.multiply(numpy.multiply(Z[:,0], Y[:,1]), X[:,2])
		ret -= numpy.multiply(numpy.multiply(Y[:,0], X[:,1]), Z[:,2])
		ret -= numpy.multiply(numpy.multiply(X[:,0], Z[:,1]), Y[:,2])
		return ret



	# One generalized winding number per input vertex
	ret = numpy.zeros(X.shape[0], dtype = X.dtype)
	
	# Accumulate generalized winding number for each triangle
	for U, V, W in triangles:	
		A, B, C = U - X, V - X, W - X
		omega = adet(A, B, C)

		a, b, c = anorm2(A), anorm2(B), anorm2(C)
		k  = a * b * c 
		k += c * numpy.sum(numpy.multiply(A, B), axis = 1)
		k += a * numpy.sum(numpy.multiply(B, C), axis = 1)
		k += b * numpy.sum(numpy.multiply(C, A), axis = 1)

		ret += numpy.arctan2(omega, k)

	# Job done
	return ret >= 2 * numpy.pi 

#import bpy

# def point_cloud(ob_name, coords, edges=[], faces=[]):
#     """Create point cloud object based on given coordinates and name.
#
#     Keyword arguments:
#     ob_name -- new object name
#     coords -- float triplets eg: [(-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)]
#     """
#
#     # Create new mesh and a new object
#     me = bpy.data.meshes.new(ob_name + "Mesh")
#     ob = bpy.data.objects.new(ob_name, me)
#
#     # Make a mesh from a list of vertices/edges/faces
#     me.from_pydata(coords, edges, faces)
#
#     # Display name and update the mesh
#     ob.show_name = True
#     me.update()
#     return ob

# Create the object
#pc = point_cloud("point-cloud", [(0.0, 0.0, 0.0)])

# Link object to the active collection
#bpy.context.collection.objects.link(pc)


def get_grid(vertices, grid_dist = 1):
	min_corner = vertices.min(axis=0)
	max_corner = vertices.max(axis=0)

	x_range = numpy.linspace(min_corner[0], max_corner[0], int((max_corner[0] - min_corner[0]) / grid_dist) + 1)
	y_range = numpy.linspace(min_corner[1], max_corner[1], int((max_corner[1] - min_corner[1]) / grid_dist) + 1)
	z_range = numpy.linspace(min_corner[2], max_corner[2], int((max_corner[2] - min_corner[2]) / grid_dist) + 1)

	grid = numpy.meshgrid(x_range, y_range, z_range)
	grid = numpy.vstack(list(map(numpy.ravel, grid))).T

	return grid