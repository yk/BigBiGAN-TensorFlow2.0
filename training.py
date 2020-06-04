from pathlib import Path
import numpy as np
import tensorflow as tf
import logging
import time
from losses import disc_loss, gen_en_loss
from misc import get_fixed_random, generate_images
import tqdm
import tensorflow_gan.examples.mnist.util as tfgan_mnist
import tensorflow_gan.examples.cifar.util as tfgan_cifar
import itertools as itt

def train(config, gen, disc_f, disc_h, disc_j, model_en, train_data, test_data, train_data_repeat, model_copies=None):
    if config.eval_metrics:

        @tf.function
        def get_mnist_eval_metrics(real, fake):
            frechet = tfgan_mnist.mnist_frechet_distance(real, fake, 1)
            score = tfgan_mnist.mnist_score(fake, 1)
            return tf.stack(list(map(tf.stop_gradient, (frechet, score))))

        @tf.function
        def get_cifar10_eval_metrics(real, fake):
            frechet = tfgan_cifar.get_frechet_inception_distance(real, fake, config.train_batch_size, config.train_batch_size)
            score = tfgan_cifar.get_inception_scores(fake, config.train_batch_size, config.train_batch_size)
            return tf.stack(list(map(tf.stop_gradient, (frechet, score))))

    if config.load_model or config.do_eval:
        if config.load_from_further:
            config_dir = Path('~/models/dg_further/configs').expanduser()
        else:
            config_dir = Path('~/models/dg/exp/configs').expanduser()
        for cdir in config_dir.iterdir():
            ldir = cdir / 'logs'
            try:
                cmdline = (ldir / 'cmdlog.txt').read_text()
                cparams = dict(k[2:].split('=') for k in cmdline.split() if k.startswith('--'))
                print(cparams)
                if (
                        cparams['dataset'] == config.dataset and
                        cparams['train_dg'] == str(config.train_dg) and
                        cparams['steps_dg'] == str(config.steps_dg) and
                        cparams['conditional'] == str(config.conditional)
                        ):
                    checkpoint_dir = list(ldir.glob('*/checkpoints'))[0]
                    break
            except:
                pass
        else:
            raise ValueError('Model not found')

    train_data_iterator = iter(train_data_repeat)
    image, label = next(train_data_iterator)

    if config.train_dg or config.do_eval or config.load_model:
        forward_pass_fn_d_const = get_forward_pass_fn()
        forward_pass_fn_g_const = get_forward_pass_fn()
        forward_pass_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], config.train_batch_size, config.num_cont_noise, config, update_d=False)
        forward_pass_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, config.train_batch_size, config.num_cont_noise, config, update_g=False)
    else:
        forward_pass_fn_d_const = get_forward_pass_fn()
        forward_pass_fn_d_const(image, label, gen, disc_f, disc_h, disc_j, model_en, config.train_batch_size, config.num_cont_noise, config)
        forward_pass_fn_g_const = None

    # Start training
    # Define optimizers
    disc_optimizer = tf.optimizers.Adam(learning_rate=config.lr_disc,
                                        beta_1=config.beta_1_disc,
                                        beta_2=config.beta_2_disc)

    gen_en_optimizer = tf.optimizers.Adam(learning_rate=config.lr_gen_en,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)

    dg_optimizer_d = tf.optimizers.Adam(learning_rate=config.lr_dg,
                                        beta_1=config.beta_1_disc,
                                       beta_2=config.beta_2_disc)

    dg_optimizer_g = tf.optimizers.Adam(learning_rate=config.lr_dg,
                                        beta_1=config.beta_1_gen_en,
                                       beta_2=config.beta_2_gen_en)


    out_dir = f'{config.result_path}/{config.model}_{config.dataset}_{time.strftime("%Y-%m-%d--%H-%M-%S")}'
    # Define Logging to Tensorboard
    summary_writer = tf.summary.create_file_writer(out_dir)

    fixed_z, fixed_c = get_fixed_random(config, num_to_generate=100)  # fixed_noise is just used for visualization.

    # Define metric
    metric_loss_gen_en = tf.keras.metrics.Mean()
    metric_loss_disc = tf.keras.metrics.Mean()
    metric_loss_dg = tf.keras.metrics.Mean()

    if config.train_dg or config.do_eval or config.load_model:
        train_step_fn_d_const = get_train_step_fn(forward_pass_fn_d_const)
        train_step_fn_g_const = get_train_step_fn(forward_pass_fn_g_const)
        train_step_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config, update_d=False)
        train_step_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config, update_g=False)
    else:
        train_step_fn_d_const = get_train_step_fn(forward_pass_fn_d_const)
        train_step_fn_d_const(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, config.train_batch_size, config.num_cont_noise, config)
        train_step_fn_g_const = None

    ckpt = tf.train.Checkpoint(gen=gen, disc_f=disc_f, disc_h=disc_h, disc_j=disc_j, model_en=model_en)
    if config.save_model:
        ckpt_mgr = tf.train.CheckpointManager(ckpt, out_dir + '/checkpoints/', config.num_epochs+1)
        ckpt_mgr.save(0)
    if config.load_model and not config.do_eval:
        ckpt.restore(tf.train.latest_checkpoint(str(checkpoint_dir))).assert_consumed()

    # Start training
    epoch_tf = tf.Variable(0, trainable=False, dtype=tf.float32)
    for epoch in range(0, config.num_epochs + config.do_eval):
        logging.info(f'Start epoch {epoch+1} ...')  # logs a message.
        epoch_tf.assign(epoch)
        start_time = time.time()

        if config.do_eval:
            ckpt.restore(str(checkpoint_dir) + f'/ckpt-{epoch}').assert_consumed()

        datasets = [ (train_data, 'train') ]
        if config.do_eval:
            datasets.append((test_data, 'test'))
        for dataset, dataset_name in datasets:
            if config.eval_metrics:
                if config.only_eval_last:
                    if epoch < config.num_epochs:
                        continue
                num_points_eval = 10000 if config.only_eval_last else 2000
                if config.debug:
                    num_points_eval = 100
                eval_metrics_batches = num_points_eval // config.train_batch_size + 1
                num_points_eval = eval_metrics_batches * config.train_batch_size
                real_images = []
                fake_images = []
                frechet = []
                score = []
                for batch in tqdm.tqdm(itt.islice(dataset, 0, eval_metrics_batches), total=eval_metrics_batches):
                    z, c = get_fixed_random(config, num_to_generate=config.train_batch_size)
                    fake = generate_images(gen, z, c, config, do_plot=False)
                    real = batch[0]
                    if len(real_images) < 200:
                        real_images.extend([np.array(i) for i in real])
                        fake_images.extend([np.array(i) for i in fake])
                    if config.dataset == 'mnist':
                        real, fake = map(lambda x: tf.image.resize(x, (28, 28)), (real, fake))
                        f, s = get_mnist_eval_metrics(real, fake)
                        frechet.append(f)
                        score.append(s)
                    elif config.dataset == 'cifar10':
                        f, s = get_cifar10_eval_metrics(real, fake)
                        frechet.append(f)
                        score.append(s)
                    elif config.dataset == 'fashion_mnist':
                        real, fake = map(lambda x: tf.concat([x]*3, -1), (real, fake))
                        f, s = get_cifar10_eval_metrics(real, fake)
                        frechet.append(f)
                        score.append(s)

                frechet = sum(frechet) / len(frechet)
                score = sum(score) / len(score)

                with summary_writer.as_default():
                    tf.summary.scalar(f'Frechet Distance ({dataset_name})',frechet,step=epoch)
                    tf.summary.scalar(f'Score ({dataset_name})', score,step=epoch)

                real_images, fake_images = map(np.array, (real_images, fake_images))
                np.save(f'logs/real-{epoch}.npy', real_images)
                np.save(f'logs/fake-{epoch}.npy', fake_images)

                continue
        
            train_epoch(train_step_fn_d_const, train_step_fn_g_const, forward_pass_fn_d_const, forward_pass_fn_g_const, train_data, train_data_iterator, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, dg_optimizer_g, dg_optimizer_d, metric_loss_disc, metric_loss_gen_en, metric_loss_dg, config.train_batch_size, config.num_cont_noise, config, model_copies=model_copies)
            epoch_time = time.time()-start_time

            # Save results
            logging.info(f'{dataset_name} - Epoch {epoch+1}: Disc_loss: {metric_loss_disc.result()}, Gen_loss: {metric_loss_gen_en.result()}, Time: {epoch_time}, DG: {metric_loss_dg.result()}')
            with summary_writer.as_default():
                tf.summary.scalar(f'Generator and Encoder loss ({dataset_name})',metric_loss_gen_en.result(),step=epoch)
                tf.summary.scalar(f'Discriminator loss ({dataset_name})', metric_loss_disc.result(),step=epoch)
                tf.summary.scalar(f'Duality Gap ({dataset_name})',metric_loss_dg.result(),step=epoch)

            metric_loss_gen_en.reset_states()
            metric_loss_dg.reset_states()

            metric_loss_disc.reset_states()
            # Generate images
            gen_image = generate_images(gen, fixed_z, fixed_c, config)
            with summary_writer.as_default():
                tf.summary.image('Generated Images', tf.expand_dims(gen_image,axis=0),step=epoch)
        if config.save_model:
            ckpt_mgr.save(epoch+1)


def make_optimizer(lr, beta1, beta2):
    optimizer = tf.optimizers.Adam(learning_rate=lr,
                                    beta_1=beta1,
                                    beta_2=beta2)
    return optimizer


def train_epoch(train_step_fn_d_const, train_step_fn_g_const, forward_pass_fn_d_const, forward_pass_fn_g_const, train_data, train_data_iterator, gen,disc_f, disc_h, disc_j, model_en, disc_optimizer,gen_en_optimizer, dg_optimizer_g, dg_optimizer_d,
                metric_loss_disc, metric_loss_gen_en, metric_loss_dg, batch_size, cont_dim, config, model_copies):
    all_models = gen, disc_f, disc_h, disc_j, model_en
    d_vars = disc_f.trainable_variables+disc_h.trainable_variables+disc_j.trainable_variables
    g_vars = gen.trainable_variables + model_en.trainable_variables
    def copy_vars(all_models=all_models):
        for m in all_models:
            for v, vc in zip(m.variables, model_copies[m].variables):
                vc.assign(v)
        if dg_optimizer_g.get_weights() and not config.do_eval:
            disc_optimizer.set_weights(dg_optimizer_d.get_weights())
            gen_en_optimizer.set_weights(dg_optimizer_g.get_weights())
    for outer_step, (image, label) in enumerate(tqdm.tqdm(train_data)):
        if config.train_dg or config.do_eval or config.load_model:
            if outer_step == 0 or not config.do_eval:
                copy_vars([gen, model_en])
                steps_dg = config.steps_dg_eval if config.do_eval else config.steps_dg
                for _ in range(steps_dg):
                    inner_image, inner_label = next(train_data_iterator)
                    train_step_fn_d_const(inner_image, inner_label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config, update_d=False)
                copy_vars([disc_f, disc_h, disc_j])
                for _ in range(steps_dg):
                    inner_image, inner_label = next(train_data_iterator)
                    train_step_fn_g_const(inner_image, inner_label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config, update_g=False)

            with tf.GradientTape() as tape:
                g_e_loss, d_loss_d_const = forward_pass_fn_d_const(image, label, model_copies[gen], disc_f, disc_h, disc_j, model_copies[model_en], batch_size, cont_dim, config)
                grad_for_d = tape.gradient(d_loss_d_const, d_vars)

            with tf.GradientTape() as tape:
                g_e_loss, d_loss_g_const = forward_pass_fn_g_const(image, label, gen, model_copies[disc_f], model_copies[disc_h], model_copies[disc_j], model_en, batch_size, cont_dim, config)
                grad_for_g = [-g for g in tape.gradient(d_loss_g_const, g_vars)]
            if not config.do_eval:
                dg_optimizer_d.apply_gradients(zip(grad_for_d, d_vars))
                dg_optimizer_g.apply_gradients(zip(grad_for_g, g_vars))
            metric_loss_dg.update_state(d_loss_d_const - d_loss_g_const)
        else:
            train_step_fn_d_const(image, label, gen, disc_f, disc_h, disc_j, model_en, disc_optimizer, gen_en_optimizer, metric_loss_disc, metric_loss_gen_en, batch_size, cont_dim, config)

        if config.debug:
            break


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




