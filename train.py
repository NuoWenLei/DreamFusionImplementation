from constants import TRAIN_STEPS, TRAIN_EPOCHS, H, W, NUM_SAMPLES, LOSS_WEIGHT
from imports import tf, np, tqdm, plt, Image, ImageDraw
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

def helper_blob(im_X, im_Y, r, fill_value):
	mask_im = Image.new("RGB", (im_X, im_Y), (0, 0, 0))
	draw = ImageDraw.Draw(mask_im)

	r = int(r)

	X = int(im_X / 2)
	Y = int(im_Y / 2)

	fill_val = int(255. * fill_value)

	draw.ellipse([(X-r, Y-r), (X+r, Y+r)], fill = (fill_val, fill_val, fill_val))

	return (np.array(mask_im).astype("float32") / 255.)[np.newaxis, ...]

# # From scikit-learn
# def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2,
#                  volume_fraction=0.5, seed=None):
#     """
#     Generate synthetic binary image with several rounded blob-like objects.
#     Parameters
#     ----------
#     length : int, optional
#         Linear size of output image.
#     blob_size_fraction : float, optional
#         Typical linear size of blob, as a fraction of ``length``, should be
#         smaller than 1.
#     n_dim : int, optional
#         Number of dimensions of output image.
#     volume_fraction : float, default 0.5
#         Fraction of image pixels covered by the blobs (where the output is 1).
#         Should be in [0, 1].
#     seed : {None, int, `numpy.random.Generator`}, optional
#         If `seed` is None the `numpy.random.Generator` singleton is used.
#         If `seed` is an int, a new ``Generator`` instance is used,
#         seeded with `seed`.
#         If `seed` is already a ``Generator`` instance then that instance is
#         used.
#     Returns
#     -------
#     blobs : ndarray of bools
#         Output binary image
#     Examples
#     --------
#     >>> from skimage import data
#     >>> data.binary_blobs(length=5, blob_size_fraction=0.2)  # doctest: +SKIP
#     array([[ True, False,  True,  True,  True],
#            [ True,  True,  True, False,  True],
#            [False,  True, False,  True,  True],
#            [ True, False, False,  True,  True],
#            [ True, False, False, False,  True]])
#     >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)
#     >>> # Finer structures
#     >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)
#     >>> # Blobs cover a smaller volume fraction of the image
#     >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)
#     """
#     # filters is quite an expensive import since it imports all of scipy.signal
#     # We lazy import here

#     rs = np.random.default_rng(seed)
#     shape = tuple([length] * n_dim)
#     mask = np.zeros(shape)
#     n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
#     points = (length * rs.random((n_dim, n_pts))).astype(int)
#     mask[tuple(indices for indices in points)] = 1
#     mask = gaussian_filter(mask, sigma=0.25 * length * blob_size_fraction)
#     threshold = np.percentile(mask, 100 * (1 - volume_fraction))
#     return np.logical_not(mask < threshold)

def train(model):
	c2w = sample_random_c2w()
	ray_origins, ray_dirs = get_rays(H, W, model.focal, c2w)
	for epoch in range(TRAIN_EPOCHS):
		print(f"Epoch: {epoch} / {TRAIN_EPOCHS}")
		
		for step in tqdm(range(TRAIN_STEPS)):
			
			rays_flat, t_vals, rays_flat_unencoded = render_flat_rays(
				ray_origins, ray_dirs, near=2.0, far=6.0, num_samples=NUM_SAMPLES, rand=False
			)
			with tf.GradientTape() as tape:
				image_observation, density_normals = model([
					rays_flat[tf.newaxis, ...], t_vals[tf.newaxis, ...], ray_origins[0, 0], rays_flat_unencoded,
					np.array([0., 0., 0.]), np.array([1., 1., 1.]) 
				])

				blob = helper_blob(W, H, H // 5, 0.1)

				image_observation = image_observation + blob
				# mean_observation = tf.reduce_mean(image_observation)

				# image_observation = image_observation - mean_observation
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

			

