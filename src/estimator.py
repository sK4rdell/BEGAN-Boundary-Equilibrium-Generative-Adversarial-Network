import os
import glob
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
from src.network_modules import Encoder, Decoder, AutoEncoder, Losses
from src.data_process.datasets import input_function



# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)
# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./train_logs',
    docstring='Output directory for model and training stats.')

tf.app.flags.DEFINE_string(
    flag_name='data_dir', default_value='./tfrecord_data',
    docstring='Directory containing the training data')

tf.app.flags.DEFINE_integer(flag_name='save_freq', default_value=500,
                            docstring='how often to store checkpoints')

tf.app.flags.DEFINE_integer(flag_name='out_size', default_value=64,
                            docstring='width and height of generated images')

tf.app.flags.DEFINE_integer(flag_name='depth', default_value=128,
                            docstring='depth of feature maps')

tf.app.flags.DEFINE_integer(flag_name='latent_size', default_value=64,
                            docstring='size of latent vector')

tf.app.flags.DEFINE_integer(flag_name='batch_size', default_value=16,
                            docstring='batch size')

tf.app.flags.DEFINE_float(
    flag_name='lr', default_value=5 * 1e-5, docstring='learning rate')


def model_architecture(x, z, is_training, params):
    """Defines the architecture of the model

    Parameters:
    ----------
    x : Tensor
        Features
    z : Tensor
        Latent vector
    is_training : bool
        true if training
    params : HParams
        hyperparameters
    Returns
    -------
     generated image, reconstructed generated image, reconsructed image,
     proportional gain, generated demo images
    """

    generator = Decoder(params.depth, 'Generator', params.out_dim)
    # generate image
    generated_img = generator(z, is_training)

    z_demo = np.random.uniform(-1., 1., size=[8, FLAGS.latent_size])
    z_demo = tf.convert_to_tensor(z_demo, dtype=tf.float32)

    gen_demo = generator(z_demo, training=False, reuse=True)
    # with tf.device('/device:GPU:0'):  # 1080

    discriminator = AutoEncoder(params.depth, 'Discriminator', params.out_dim,
                                params.latent_size)

    # reconstruct real image
    rec_img = discriminator(x, is_training)
    # reconstruct generated image
    rec_gen_img = discriminator(generated_img, is_training, reuse=True)

    k = tf.Variable(initial_value=0, dtype=tf.float32, trainable=True)

    return generated_img, rec_gen_img, rec_img, k, gen_demo


def get_train_op_fn(loss, params, variables):
    """ Returns the train op.

    Parameters:
    ----------
    loss : Tensor
        loss-function to optimize
    params : HParams
        hyper-parameters
    variables : list
        set of trainable variables to optimize
    Returns
    -------
        train-op function
    """

    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.lr,
        variables=variables
    )


def denormalize_img(img):
    """ denormalizes the image. 

    Parameters:
    ----------
    img : Tensor
        normalized image
    """

    return (img + .5) * 255


def model_fn(features, labels, mode, params):
    """ Defines the Estimator specs.

    Parameters:
    ----------
    features : Tensor
        real images
    labels : Tensor
        latent vector z
    mode : Mode
        training mode
    params : HParams
        hyper-parameters
    Returns
    -------
    EstimatorSpec
    """
    features = tf.placeholder_with_default(
        features, shape=[None, 64, 64, 3], name='img_input')

    is_training = mode == ModeKeys.TRAIN
    z = labels
    gen_img, rec_gen_img, rec_img, k, gen_demo = model_architecture(
        features, z, is_training, params)

    if mode != ModeKeys.INFER:

        generator_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        discriminator_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        # calculate losses for generatore and discriminator
        gen_loss, disc_loss, m, k_op = Losses.BEGAN_loss(
            features, rec_img, gen_img, rec_gen_img, k)
        # generator train-op
        gen_train_op = get_train_op_fn(gen_loss, params, generator_vars)
        # discriminator train-op
        disc_train_op = get_train_op_fn(disc_loss, params, discriminator_vars)
        # combine train-ops
        train_op = tf.group(gen_train_op, disc_train_op, k_op)
        loss = tf.group(gen_loss, disc_loss)
    else:
        loss, train_op = None, None

    # denormalize image
    generated_images = denormalize_img(gen_img)
    # summaries
    tf.summary.image("Generated_images", generated_images, max_outputs=6)
    tf.summary.image("Generated_img_fixed_z", gen_demo, max_outputs=8)
    tf.summary.scalar("Generator_ls", gen_loss)
    tf.summary.scalar("k", k)
    tf.summary.scalar("convergence_meassure_M", m)
    tf.summary.scalar("Discriminator_loss", disc_loss)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=gen_loss,
        train_op=train_op)


def get_estimator(run_config, params):
    """Returns the Estimator. 

    Parameters:
    ----------
    run_config : 
        runtime configs
    params : {[type]}
        hyperparameters
    Returns
    -------
    Estimator
    """

    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # params
        config=run_config  # RunConfig
    )


def experiment_fn(run_config, params):
    """Create an experiment to train the model.

    Parameters:
    ----------
    run_config : RunConfig
        Configuration for Estimator run
    params : HParam
        Hyperparameters 
    Returns
    -------
    Experiment
        Experiment for trainignt the model
    """

    # You can change a subset of the run_config properties as
    run_config = run_config.replace(
        save_checkpoints_steps=FLAGS.save_freq)

    estimator = get_estimator(run_config, params)

    # maybe create data_dir
    os.makedirs(FLAGS.data_dir, exist_ok=True)
    train_files = glob.glob(FLAGS.data_dir + '/*training*')
    train_input_fn = input_function(
        train_files, batch_size=params.batch_size, epochs=None, z_size=params.latent_size)
    eval_input_fn = train_input_fn  # never used
    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=0,  # Eval frequency
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


def run_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters

    params = tf.contrib.training.HParams(lr=FLAGS.lr,
                                         train_steps=None,
                                         out_dim=FLAGS.out_size,
                                         depth=128,
                                         latent_size=FLAGS.latent_size,
                                         batch_size=FLAGS.batch_size)
    # Set the run_config and the directory to save the model and stats
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    run_config = tf.contrib.learn.RunConfig(session_config=config,
                                            save_checkpoints_secs=30 * 60,
                                            keep_checkpoint_max=70)
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params
    )


if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
