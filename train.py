from constants import TRAIN_STEPS, TRAIN_EPOCHS, H, W, NUM_SAMPLES, LOSS_WEIGHT
from imports import tf, np, tqdm, plt
from NeRF_camera import sample_random_c2w
from NeRF_rays import get_rays, render_flat_rays
from stable_diffusion_loss import diffuse_loss

def make_plt(im, epoch):
	plt.figure()
	plt.imshow(tf.keras.preprocessing.image.array_to_img(im))
	plt.set_title(f"Predicted Image: {epoch:03d}")
	plt.show()

def train(model):
	for epoch in range(TRAIN_EPOCHS):
		print(f"Epoch: {epoch} / {TRAIN_EPOCHS}")
		c2w = sample_random_c2w()
		for step in tqdm(range(TRAIN_STEPS)):
			ray_origins, ray_dirs = get_rays(H, W, model.focal, c2w)
			rays_flat, t_vals = render_flat_rays(
				ray_origins, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
			)
			with tf.GradientTape() as tape:
				image_observation = model([
					rays_flat[tf.newaxis, ...], t_vals[tf.newaxis, ...], ray_origins[0],
					np.array([0., 0., 0.]), np.array([1., 1., 1.]) 
				])
				pred_et, true_et = diffuse_loss(model.diffuse_model, model.target_text, image_observation)
				loss = LOSS_WEIGHT * tf.reduce_sum(tf.stop_gradient(pred_et - true_et) * image_observation)

			trainable_params = model.nerf_model.trainable_variables
			gradients = tape.gradient(loss, trainable_params)
			model.optimizer.apply_gradients(zip(gradients, trainable_params))
			if step % 50 == 0:
				print(tf.shape(image_observation))
		make_plt(image_observation, epoch)

			

