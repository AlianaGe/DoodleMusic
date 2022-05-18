"""This script performs inference from a trained model."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml, update_not_none

LOGGER = logging.getLogger("musegan.inference")
PARAMS_PATH = ''

now_path = r'D:\MyFiles\FinalTest\music\AI-music'
#def setup(params_path=os.path.join(os.getcwd(), 'params.yaml'), config_path=os.path.join(os.getcwd(), 'config.yaml')):

def setup(params_path=os.path.join(now_path, 'params.yaml'), config_path=os.path.join(now_path, 'config.yaml')):
    """Parse command line arguments, load model parameters, load configurations
    and setup environment."""

    # Load parameters
    params = load_yaml(params_path)

    # Load training configurations
    config = load_yaml(config_path)
    # update_not_none(config, vars(args))

    # Set unspecified schedule steps to default values
    for target in (config['learning_rate_schedule'], config['slope_schedule']):
        if target['start'] is None:
            target['start'] = 0
        if target['end'] is None:
            target['end'] = config['steps']

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    return params, config


def musegan(condition_track, result_dir, suffix_idx):
    # Setup
    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    params, config = setup()
    # ============================== Placeholders ==============================
    placeholder_z = tf.placeholder(
        tf.float32, shape=(None, params['latent_dim']))
    placeholder_c = tf.placeholder(
        tf.float32, shape=([None] + params['data_shape'][:-1] + [1]))
    placeholder_suffix = tf.placeholder(tf.string)

    # ================================= Model ==================================
    # Build model
    model = Model(params)
    predict_nodes = model(
        c=placeholder_c, z=placeholder_z, mode='predict', params=params)

    # ========================== Session Preparation ===========================
    # Get tensorflow session config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # Create saver to restore variables
    saver = tf.train.Saver()

    # =========================== Tensorflow Session ===========================
    with tf.Session(config=tf_config) as sess:
        # Restore the latest checkpoint
        LOGGER.info("Restoring the latest checkpoint.")
        with open(os.path.join(config['checkpoint_dir'], 'checkpoint')) as f:
            checkpoint_name = os.path.basename(
                f.readline().split()[1].strip('"'))
        checkpoint_path = os.path.realpath(
            os.path.join(config['checkpoint_dir'], checkpoint_name))
        saver.restore(sess, checkpoint_path)

        # Run sampler op
        feed_dict_sampler = {
            placeholder_z: scipy.stats.truncnorm.rvs(
                config['lower'], config['upper'], size=(
                    (config['rows'] * config['columns']),
                    params['latent_dim'])),
            placeholder_suffix: suffix_idx,
            placeholder_c: np.array([condition_track])}
        return sess.run(predict_nodes['fake_x'],
                        feed_dict=feed_dict_sampler)  # run(self, fetches, feed_dict=None, options=None, run_metadata=None)
