import tensorflow as tf
import logging
import time
from losses import disc_loss, gen_en_loss
from misc import get_fixed_random, generate_images
import tqdm

def train(config, gen, disc_f, disc_h, disc_j, model_en, train_data, train_data_repeat, model_copies=None):
    train_data_iterator = iter(train_data_repeat)
    image, label = next(train_data_iterator)
    forward_pass_fn_d_const = get_forward_pass_fn()
    forward_pass_fn_g_const = get_forward_pass_fn()
    forward_pass_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], config.train_batch_size, config.num_cont_noise, config, update_d=False)
    forward_pass_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, config.train_batch_size, config.num_cont_noise, config, update_g=False)

    # Start training
    # Define optimizers
    disc_optimizer = tf.optimizers.Adam(learning_rate=config.lr_disc,
                                        beta_1=config.beta_1_disc,
                                        beta_2=config.beta_2_disc)

    gen_en_optimizer = tf.optimizers.Adam(learning_rate=config.lr_gen_en,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)

    dg_optimizer = tf.optimizers.Adam(learning_rate=config.lr_dg,
                                        beta_1=config.beta_1_disc,
                                       beta_2=config.beta_2_disc)


    # Define Logging to Tensorboard
    summary_writer = tf.summary.create_file_writer(f'{config.result_path}/{config.model}_{config.dataset}_{time.strftime("%Y-%m-%d--%H-%M-%S")}')

    fixed_z, fixed_c = get_fixed_random(config, num_to_generate=100)  # fixed_noise is just used for visualization.

    # Define metric
    metric_loss_gen_en = tf.keras.metrics.Mean()
    metric_loss_disc = tf.keras.metrics.Mean()
    metric_loss_dg = tf.keras.metrics.Mean()

    train_step_fn_d_const = get_train_step_fn(forward_pass_fn_d_const)
    train_step_fn_g_const = get_train_step_fn(forward_pass_fn_g_const)
    train_step_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config, update_d=False)
    train_step_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config, update_g=False)

    # Start training
    epoch_tf = tf.Variable(0, trainable=False, dtype=tf.float32)
    for epoch in range(config.num_epochs):
        logging.info(f'Start epoch {epoch+1} ...')  # logs a message.
        epoch_tf.assign(epoch)
        start_time = time.time()

        train_epoch(train_step_fn_d_const, train_step_fn_g_const, forward_pass_fn_d_const, forward_pass_fn_g_const, train_data, train_data_iterator, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, dg_optimizer, metric_loss_disc,
                    metric_loss_gen_en, metric_loss_dg, config.train_batch_size, config.num_cont_noise, config, model_copies=model_copies)
        epoch_time = time.time()-start_time

        # Save results
        logging.info(f'Epoch {epoch+1}: Disc_loss: {metric_loss_disc.result()}, Gen_loss: {metric_loss_gen_en.result()}, Time: {epoch_time}, DG: {metric_loss_dg.result()}')
        with summary_writer.as_default():
            tf.summary.scalar('Generator and Encoder loss',metric_loss_gen_en.result(),step=epoch)
            tf.summary.scalar('Discriminator loss', metric_loss_disc.result(),step=epoch)
            tf.summary.scalar('Duality Gap',metric_loss_dg.result(),step=epoch)

        metric_loss_gen_en.reset_states()
        metric_loss_dg.reset_states()

        metric_loss_disc.reset_states()
        # Generate images
        gen_image = generate_images(gen, fixed_z, fixed_c, config)
        with summary_writer.as_default():
            tf.summary.image('Generated Images', tf.expand_dims(gen_image,axis=0),step=epoch)


def make_optimizer(lr, beta1, beta2):
    optimizer = tf.optimizers.Adam(learning_rate=lr,
                                    beta_1=beta1,
                                    beta_2=beta2)
    return optimizer


def train_epoch(train_step_fn_d_const, train_step_fn_g_const, forward_pass_fn_d_const, forward_pass_fn_g_const, train_data, train_data_iterator, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer, dg_optimizer,
                metric_loss_disc, metric_loss_gen_en, metric_loss_dg, batch_size, cont_dim, config, model_copies):
    all_models = gen, disc_f, disc_h, disc_j, model_en
    d_vars = disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables
    g_vars = gen.trainable_variables + model_en.trainable_variables
    def copy_vars():
        for m in all_models:
            for v, vc in zip(m.variables, model_copies[m].variables):
                vc.assign(v)
        dgw = dg_optimizer.get_weights()
        if dgw:
            num_d_weights = 1 + 2 * len(d_vars)
            disc_optimizer.set_weights(dgw[:num_d_weights])
            gen_en_optimizer.set_weights(dgw[num_d_weights:])
    for image, label in tqdm.tqdm(train_data):
        copy_vars()
        for _ in range(config.steps_dg):
            inner_image, inner_label = next(train_data_iterator)
            train_step_fn_d_const(inner_image, inner_label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config, update_d=False)
        with tf.GradientTape() as tape:
            g_e_loss, d_loss = forward_pass_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], batch_size, cont_dim, config)
            grad_for_d = tape.gradient(d_loss, d_vars)

        copy_vars()
        for _ in range(config.steps_dg):
            inner_image, inner_label = next(train_data_iterator)
            train_step_fn_g_const(inner_image, inner_label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config, update_g=False)
        with tf.GradientTape() as tape:
            g_e_loss, d_loss = forward_pass_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, batch_size, cont_dim, config)
            grad_for_g = [-g for g in tape.gradient(d_loss, g_vars)]
        dg_optimizer.apply_gradients(zip(grad_for_d+grad_for_g, d_vars+g_vars))


def get_forward_pass_fn():
    @tf.function
    def forward_pass(image, label, gen, disc_f, disc_h, disc_j, model_en, batch_size, cont_dim, config, update_g=True, update_d=True):
        if not config.conditional:
            label = None
        fake_noise = tf.random.truncated_normal([batch_size, cont_dim])
        fake_img = gen(fake_noise, label, training=True)
        latent_code_real = model_en(image, training=True)
        real_f_to_j, real_f_score = disc_f(image, label, training=True)
        fake_f_to_j, fake_f_score = disc_f(fake_img, label, training=True)
        real_h_to_j, real_h_score = disc_h(latent_code_real, training=True)
        fake_h_to_j, fake_h_score = disc_h(fake_noise, training=True)
        real_j_score = disc_j(real_f_to_j, real_h_to_j, training=True)
        fake_j_score = disc_j(fake_f_to_j, fake_h_to_j, training=True)

        d_loss = disc_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
        g_e_loss = gen_en_loss(real_f_score, real_h_score, real_j_score, fake_f_score, fake_h_score, fake_j_score)
        return g_e_loss, d_loss
    return forward_pass


def get_train_step_fn(forward_pass_fn):
    @tf.function
    def train_step(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc,
                metric_loss_gen_en, batch_size, cont_dim, config, update_g=True, update_d=True):

        with tf.device('{}:*'.format(config.device)):
            for _ in range(config.D_G_ratio):
                with tf.GradientTape() as gen_en_tape, tf.GradientTape() as disc_tape:
                    g_e_loss, d_loss = forward_pass_fn(image, label, gen, disc_f, disc_h, disc_j, model_en, batch_size, cont_dim, config, update_g=update_g, update_d=update_d)

                if update_d:
                    grad_disc = disc_tape.gradient(d_loss, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables)

                    disc_optimizer.apply_gradients(zip(grad_disc, disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables))
                metric_loss_disc.update_state(d_loss)  # upgrade the value in metrics for single step.

            if update_g:
                grad_gen_en = gen_en_tape.gradient(g_e_loss, gen.trainable_variables + model_en.trainable_variables)

                gen_en_optimizer.apply_gradients(zip(grad_gen_en, gen.trainable_variables + model_en.trainable_variables))
            metric_loss_gen_en.update_state(g_e_loss)

            del gen_en_tape
            del disc_tape
    return train_step




