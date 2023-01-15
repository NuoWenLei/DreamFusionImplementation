from imports import tf, np, tqdm, _UNCONDITIONAL_TOKENS, math

def get_context_from_prompt(model, prompt, batch_size):
	inputs = model.tokenizer.encode(prompt)
	assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
	phrase = inputs + [49407] * (77 - len(inputs))
	phrase = np.array(phrase)[None].astype("int32")
	phrase = np.repeat(phrase, batch_size, axis=0)

	# Encode prompt tokens (and their positions) into a "context vector"
	pos_ids = np.array(list(range(77)))[None].astype("int32")
	pos_ids = np.repeat(pos_ids, batch_size, axis=0)
	return model.text_encoder.predict_on_batch([phrase, pos_ids]), pos_ids

def get_model_output(
    model,
    latent,
    t,
    context,
    unconditional_context,
    unconditional_guidance_scale,
    batch_size
):
    timesteps = np.array([t])
    t_emb = model.timestep_embedding(timesteps)
    t_emb = np.repeat(t_emb, batch_size, axis=0)
    unconditional_latent = model.diffusion_model(
        [latent, t_emb, unconditional_context]
    )
    latent = model.diffusion_model([latent, t_emb, context])
    return unconditional_latent + unconditional_guidance_scale * (
        latent - unconditional_latent
    ), unconditional_latent, latent

def diffuse_loss(model, prompt, image_observation,
				unconditional_guidance_scale = 7.5,
				batch_size = 1):
	context, pos_ids = get_context_from_prompt(model, prompt, batch_size)
	unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
	unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
	model.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
	unconditional_context = model.text_encoder.predict_on_batch(
		[model.unconditional_tokens, pos_ids]
	)

	# Check range of image_observation prediction
	# print(tf.math.reduce_min(image_observation), tf.math.reduce_max(image_observation))

	min_cut_observation = (image_observation - tf.math.reduce_min(image_observation))

	scaled_observation = (min_cut_observation * 2. / tf.math.reduce_max(min_cut_observation)) - 1.

	timestep = (tf.cast(tf.random.uniform((), 0.02, 0.98) * 999., "int32")) + 1

	_, alphas, alphas_prev = model.get_starting_parameters(
		[timestep], batch_size, None
		)

	true_et = tf.random.normal(tf.shape(scaled_observation))

	a_t = alphas[0]
	sigma_t = ((1.0 - a_t) ** 0.5) # TODO: check if use alphas_prev or alphas

	diffused_observation = scaled_observation * math.sqrt(a_t) + true_et * sigma_t

	pred_et, _, _ = get_model_output(
		model,
		diffused_observation,
		timestep,
		context,
		unconditional_context,
		unconditional_guidance_scale,
		batch_size
	)
	
	return pred_et, true_et, sigma_t

def manual_encode_and_diffuse(model, prompt,
                              num_steps = 50,
                              unconditional_guidance_scale = 7.5,
                              temperature = 1,
                              batch_size = 1,
                              seed = None,
                              input_image_strength=0.5,
                              apply_et_transform=None,
                              ):
	latents = []
	e_ts = []
	conds = []
	unconds = []
	preds = []
		# Tokenize prompt (i.e. starting context)
	# inputs = model.tokenizer.encode(prompt)
	# assert len(inputs) < 77, "Prompt is too long (should be < 77 tokens)"
	# phrase = inputs + [49407] * (77 - len(inputs))
	# phrase = np.array(phrase)[None].astype("int32")
	# phrase = np.repeat(phrase, batch_size, axis=0)

	# # Encode prompt tokens (and their positions) into a "context vector"
	# pos_ids = np.array(list(range(77)))[None].astype("int32")
	# pos_ids = np.repeat(pos_ids, batch_size, axis=0)
	# context = model.text_encoder.predict_on_batch([phrase, pos_ids])

	context, pos_ids = get_context_from_prompt(model, prompt, batch_size)
	
	
	# Encode unconditional tokens (and their positions into an
	# "unconditional context vector"
	unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
	unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
	model.unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
	unconditional_context = model.text_encoder.predict_on_batch(
		[model.unconditional_tokens, pos_ids]
	)
	timesteps = np.arange(1, 1000, 1000 // num_steps)
	input_img_noise_t = timesteps[ int(len(timesteps)*input_image_strength) ]
	latent, alphas, alphas_prev = model.get_starting_parameters(
		timesteps, batch_size, seed
		)
	# if apply_et_transform is not None:
	#       latent = apply_et_transform(latent.numpy())
	#       latent = tf.constant(latent)
	# Diffusion stage
	progbar = tqdm(list(enumerate(timesteps))[::-1])
	for index, timestep in progbar:
		progbar.set_description(f"{index:3d} {timestep:3d}")
		e_t, uncond_lat, cond_lat = get_model_output(
			model,
			latent,
			timestep,
			context,
			unconditional_context,
			unconditional_guidance_scale,
			batch_size,
		)
		latents.append(latent)
		e_ts.append(e_t)
		conds.append(cond_lat)
		unconds.append(uncond_lat)
		a_t, a_prev = alphas[index], alphas_prev[index]
		latent, pred_x0 = model.get_x_prev_and_pred_x0(
			latent, e_t, index, a_t, a_prev, temperature, seed
		)
		preds.append(pred_x0)
	latents.append(latent)
	return latent, latents, preds, context, e_ts, unconds, conds

# def get_x_prev_and_pred_x0(x, e_t, a_t, a_prev, temperature, seed):
#       sigma_t = 0
#       sqrt_one_minus_at = math.sqrt(1 - a_t)
#       pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

#       # Direction pointing to x_t
#       dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
#       noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
#       x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
#       return (pred_x0 * math.sqrt(a_t) + e_t * sqrt_one_minus_at) if halt_step else x_prev, pred_x0