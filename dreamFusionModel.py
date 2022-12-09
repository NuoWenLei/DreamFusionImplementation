from imports import tf
from NeRF_MLP import render_rgb_depth
from NeRF_color import depth_to_normals, color_shading
from stable_diffusion_loss import diffuse_loss

class BudgetDreamFusion(tf.keras.models.Model):
	def __init__(self, target_text, nerf_model, diffuse_model, optimizer, loss_function = None):
		self.target_text = target_text
		self.nerf_model = nerf_model
		self.diffuse_model = diffuse_model
		self.focal = (64. * tf.random.uniform((), 70, 135) / 100.)
		self.optimizer = optimizer

	def call(self, inputs):
		rays_flat, t_vals, rays_origin, light_color, light_ambient = inputs
		albedo, depth_map = render_rgb_depth(self.nerf_model, rays_flat, t_vals, train = True)
		density_normals = depth_to_normals(depth_map)
		colored_result = color_shading(density_normals, albedo, rays_flat, rays_origin, light_color, light_ambient)
		return colored_result



