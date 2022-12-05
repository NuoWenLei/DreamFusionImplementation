from imports import tf, np, nerf_mlp_predict

def color_shading(density_gradient, albedo, mu, light_coord, light_color = np.array([0.9, 0.9, 0.9]), light_ambient = np.array([0.1, 0.1, 0.1])):
	# TODO: Are mu_points stored in a matrix and can we do batch processing?
	adjusted_light_coord = light_coord - mu
	shading = tf.reduce_sum(
		light_color * tf.nn.relu(density_gradient * adjusted_light_coord / tf.reduce_sum(adjusted_light_coord ** 2)) + light_ambient,
		axis = -1)

	# Diffuse Surface Color = p (albedo) * ((light_color * max(0, normal . light_coordinate)) + light_ambient)
	color_shading = albedo * shading
	return color_shading


def mu_point_sampling():
	pass

def pred_color_model(sampled_lambda, light_data):

	# sampled_lambda sampled from positional encoding
	light_coord, light_color, light_ambient = light_data

	# TODO: how are ray points sampled in NeRF
	mu_points = mu_point_sampling()
	densities = []
	density_normals = []
	albedos = []
	colors = []
	for mu in mu_points:
		covariance_mat = (sampled_lambda ** 2) * tf.eye(3)

		# TODO: activation function for density should be exp and albedo should be sigmoid
		result_dict = nerf_mlp_predict(0, [mu, covariance_mat])
		c = color_shading(result_dict.normal, result_dict.rgb, mu,  light_coord, light_color, light_ambient)

		densities.append(result_dict.density)
		density_normals.append(result_dict.normal)
		albedos.append(result_dict.rgb)
		colors.append(c)
	return colors