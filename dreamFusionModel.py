from imports import tf
from NeRF_MLP import render_rgb_depth
from NeRF_color import depth_to_normals, color_shading

class BudgetDreamFusion(tf.keras.models.Model):
	def __init__(self, target_text, nerf_model, diffuse_model, optimizer):
		super().__init__()
		self.target_text = target_text
		self.nerf_model = nerf_model
		self.diffuse_model = diffuse_model
		self.focal = (64. * tf.random.uniform((), 70, 135) / 100.)
		self.optimizer = optimizer

	def call(self, inputs):
		rays_flat, t_vals, rays_origin, rays_flat_unencoded, light_color, light_ambient = inputs
		albedo, depth_map = render_rgb_depth(self.nerf_model, rays_flat, t_vals, train = True)
		density_normals = depth_to_normals(depth_map[0])[tf.newaxis, ...]
		# print([tf.shape(x) for x in [density_normals, albedo, rays_flat_unencoded, rays_origin]])
		colored_result = tf.concat([albedo, tf.reduce_mean(albedo, axis = -1)[..., tf.newaxis]], axis = -1)
		# colored_result = color_shading(density_normals, albedo, rays_flat_unencoded, rays_origin, light_color, light_ambient)
	
		return colored_result



