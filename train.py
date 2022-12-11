from constants import TRAIN_STEPS, TRAIN_EPOCHS, H, W, NUM_SAMPLES, LOSS_WEIGHT
from imports import tf, np, tqdm, plt
from NeRF_camera import sample_random_c2w
from NeRF_rays import get_rays, render_flat_rays
from stable_diffusion_loss import diffuse_loss

def make_plt(im, epoch, title = None):
	plt.figure()
	plt.imshow(tf.keras.utils.array_to_img(im[0]))
	if title:
		plt.title(f"{title}: {epoch:03d}")
	else:
		plt.title(f"Predicted Image: {epoch:03d}")
	plt.show()

def train(model):
	c2w = sample_random_c2w()
	for epoch in range(TRAIN_EPOCHS):
		print(f"Epoch: {epoch} / {TRAIN_EPOCHS}")
		
		for step in tqdm(range(TRAIN_STEPS)):
			ray_origins, ray_dirs = get_rays(H, W, model.focal, c2w)
			rays_flat, t_vals, rays_flat_unencoded = render_flat_rays(
				ray_origins, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
			)
			with tf.GradientTape() as tape:
				image_observation, density_normals = model([
					rays_flat[tf.newaxis, ...], t_vals[tf.newaxis, ...], ray_origins[0, 0], rays_flat_unencoded,
					np.array([0., 0., 0.]), np.array([1., 1., 1.]) 
				])
				pred_et, true_et, sigma = diffuse_loss(model.diffuse_model, model.target_text, image_observation)
				loss = sigma * tf.reduce_sum((tf.stop_gradient(pred_et - true_et) * image_observation) ** 2)

			trainable_params = model.nerf_model.trainable_variables
			gradients = tape.gradient(loss, trainable_params)
			model.optimizer.apply_gradients(zip(gradients, trainable_params))
			if step % 10 == 0:
				print(f"Step {step}: Loss {loss}")
		make_plt(image_observation, epoch)
		make_plt(density_normals, epoch, title = "Density Normals")
		make_plt(pred_et, epoch, title = "Predicted noise")
		make_plt(true_et, epoch, title = "True noise")

			

