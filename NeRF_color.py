from imports import tf, np, convolve2d

def color_shading(density_gradient, albedo, mu, light_coord, light_color = np.array([0.9, 0.9, 0.9]), light_ambient = np.array([0.1, 0.1, 0.1])):
	# TODO: Are mu_points stored in a matrix and can we do batch processing?
	adjusted_light_coord = light_coord - mu
	shading = tf.reduce_sum(
		light_color * tf.nn.relu(density_gradient * adjusted_light_coord / (tf.reduce_sum(adjusted_light_coord ** 2)) ** 0.5) + light_ambient,
		axis = -1)

	# Diffuse Surface Color = p (albedo) * ((light_color * max(0, normal . light_coordinate)) + light_ambient)
	color_shading = albedo * shading
	return color_shading

def depth_to_normals(depth):
    """Assuming `depth` is orthographic, linearize it to a set of normals."""
    f_blur = np.array([1, 2, 1]) / 4
    f_edge = np.array([-1, 0, 1]) / 2
    dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
    dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
    inv_denom = 1 / np.sqrt(1 + dx**2 + dy**2)
    normals = np.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
    return normals

