from imports import tf
from constants import POS_ENCODE_DIMS, BATCH_SIZE, H, W, NUM_SAMPLES

def get_nerf_model(num_layers, num_pos):
    """Generates the NeRF neural network.

    Args:
        num_layers: The number of MLP layers.
        num_pos: The number of dimensions of positional encoding.

    Returns:
        The [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) model.
    """
    inputs = tf.keras.Input(shape=(num_pos, 2 * 3 * POS_ENCODE_DIMS + 3))
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Dense(units=64, activation="relu")(x)
        if i % 4 == 0 and i > 0:
            # Inject residual connection.
            x = tf.keras.layers.concatenate([x, inputs], axis=-1)
    outputs = tf.keras.layers.Dense(units=4)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def render_rgb_depth(model, rays_flat, t_vals, rand=False, train=True, early_aid_blob = None):
	"""Generates the RGB image and depth map from model prediction.

	Args:
		model: The MLP model that is trained to predict the rgb and
			volume density of the volumetric scene.
		rays_flat: The flattened rays that serve as the input to
			the NeRF model.
		t_vals: The sample points for the rays.
		rand: Choice to randomise the sampling strategy.
		train: Whether the model is in the training or testing phase.

	Returns:
		Tuple of rgb image and depth map.
	"""
	# Get the predictions from the nerf model and reshape it.
	predictions = model(rays_flat)
	predictions = tf.reshape(predictions, shape=(BATCH_SIZE, H, W, NUM_SAMPLES, 4))

	# Slice the predictions into rgb and sigma.
	rgb = tf.sigmoid(predictions[..., :-1])
	sigma_a = tf.math.exp(predictions[..., -1])
	if early_aid_blob:
		sigma_a = sigma_a + early_aid_blob
		
	# Get the distance of adjacent intervals.
	delta = t_vals[..., 1:] - t_vals[..., :-1]
	# delta shape = (num_samples)
	delta = tf.concat(
			[delta, tf.broadcast_to([1e10], shape=(BATCH_SIZE, 1))], axis=-1
		)
	alpha = 1.0 - tf.exp(-sigma_a * delta[:, None, None, :])
		

	# Get transmittance.
	exp_term = 1.0 - alpha
	epsilon = 1e-10
	transmittance = tf.math.cumprod(exp_term + epsilon, axis=-1, exclusive=True)
	weights = alpha * transmittance
	rgb = tf.reduce_sum(weights[..., None] * rgb, axis=-2)

	depth_map = tf.reduce_sum(weights * t_vals[:, None, None], axis=-1)
	return (rgb, depth_map)