import logging
import tensorflow as tf
from data import get_dataset, get_train_pipeline
from training import train
from model_small import BIGBIGAN_G, BIGBIGAN_D_F, BIGBIGAN_D_H, BIGBIGAN_D_J, BIGBIGAN_E


def set_up_train(config):
    # Setup tensorflow
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    if 'GPU' in config.device:
        physical_devices = tf.config.experimental.list_physical_devices(config.device)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    # Load dataset
    logging.info('Getting dataset...')
    train_data_raw, test_data_raw = get_dataset(config)

    # setup input pipeline
    logging.info('Generating input pipeline...')
    train_data = get_train_pipeline(train_data_raw, config)
    test_data = get_train_pipeline(test_data_raw, config)
    train_data_repeat = get_train_pipeline(train_data_raw, config, repeat=True)

    # get model
    logging.info('Prepare model for training...')
    weight_init = tf.initializers.orthogonal()
    if config.dataset == 'mnist':
        weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    model_generator = BIGBIGAN_G(config, weight_init)
    model_discriminator_f = BIGBIGAN_D_F(config, weight_init)
    model_discriminator_h = BIGBIGAN_D_H(config, weight_init)
    model_discriminator_j = BIGBIGAN_D_J(config, weight_init)
    model_encoder = BIGBIGAN_E(config, weight_init)

    model_copies = {}
    for m in (model_generator, model_discriminator_f, model_discriminator_h, model_discriminator_j, model_encoder):
        with tf.name_scope('copy'):
            model_copies[m] = m.__class__(config, weight_init)


    # train
    logging.info('Start training...')

    train(config=config,
          gen=model_generator,
          disc_f=model_discriminator_f,
          disc_h=model_discriminator_h,
          disc_j=model_discriminator_j,
          model_en=model_encoder,
          train_data=train_data, test_data=test_data, train_data_repeat=train_data_repeat, model_copies=model_copies)
    # Finished
    logging.info('Training finished ;)')
