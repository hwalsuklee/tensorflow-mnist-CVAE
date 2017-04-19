import tensorflow as tf
import numpy as np
import mnist_data
import os
import vae
import plot_utils
import glob

import argparse

IMAGE_SIZE_MNIST = 28

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Conditional Variational AutoEncoder (CVAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required = True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-3, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    parser.add_argument('--PRR', type=bool, default=True,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    parser.add_argument('--PARR', type=bool, default=False,
                        help='Boolean for plot-analogical-reasoning-result')

    parser.add_argument('--PARR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PARR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # --results_path
    try:
        os.mkdir(args.results_path)
    except(FileExistsError):
        pass
    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    # --PARR
    try:
        assert args.PARR == True or args.PARR == False
    except:
        print('PARR must be boolean type')
        return None

    if args.PARR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PARR : dim_z must be two')

        # --PARR_resize_factor
        try:
            assert args.PARR_resize_factor > 0
        except:
            print('PARR : resize factor for each displayed image must be positive')

        # --PARR_z_range
        try:
            assert args.PARR_z_range > 0
        except:
            print('PARR : range for unifomly distributed latent vector must be positive')

    return args

"""main function"""
def main(args):

    """ parameters """
    RESULTS_DIR = args.results_path

    # network architecture
    ADD_NOISE = args.add_noise

    n_hidden = args.n_hidden
    dim_img = IMAGE_SIZE_MNIST**2  # number of pixels for a MNIST image
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    # Plot
    PRR = args.PRR                              # Plot Reproduce Result
    PRR_n_img_x = args.PRR_n_img_x              # number of images along x-axis in a canvas
    PRR_n_img_y = args.PRR_n_img_y              # number of images along y-axis in a canvas
    PRR_resize_factor = args.PRR_resize_factor  # resize factor for each image in a canvas

    PMLR = args.PMLR                            # Plot Manifold Learning Result
    PMLR_n_img_x = args.PMLR_n_img_x            # number of images along x-axis in a canvas
    PMLR_n_img_y = args.PMLR_n_img_y            # number of images along y-axis in a canvas
    PMLR_resize_factor = args.PMLR_resize_factor# resize factor for each image in a canvas
    PMLR_z_range = args.PMLR_z_range            # range for random latent vector
    PMLR_n_samples = args.PMLR_n_samples        # number of labeled samples to plot a map from input data space to the latent space

    PARR = args.PARR                              # Plot Analogical Reasoning Result
    PARR_resize_factor = args.PARR_resize_factor  # resize factor for each image in a canvas
    PARR_z_range = args.PARR_z_range              # range for random latent vector

    """ prepare MNIST data """

    train_total_data, train_size, _, _, test_data, test_labels = mnist_data.prepare_MNIST_data()
    n_samples = train_size

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    y = tf.placeholder(tf.float32, shape=[None, mnist_data.NUM_LABELS], name='target_labels')

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # input for PMLR
    z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')
    fack_id_in = tf.placeholder(tf.float32, shape=[None, mnist_data.NUM_LABELS], name='latent_variable') # condition

    # network architecture
    x_, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, y, dim_img, dim_z, n_hidden, keep_prob)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """

    # Plot for reproduce performance
    if PRR:
        PRR = plot_utils.Plot_Reproduce_Performance(RESULTS_DIR, PRR_n_img_x, PRR_n_img_y, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PRR_resize_factor)

        x_PRR = test_data[0:PRR.n_tot_imgs, :]
        id_PRR = test_labels[0:PRR.n_tot_imgs, :]

        x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
        PRR.save_images(x_PRR_img, name='input.jpg')

        if ADD_NOISE:
            x_PRR = x_PRR * np.random.randint(2, size=x_PRR.shape)
            x_PRR += np.random.randint(2, size=x_PRR.shape)

            x_PRR_img = x_PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
            PRR.save_images(x_PRR_img, name='input_noise.jpg')

    # Plot for manifold learning result
    if PMLR and dim_z == 2:

        PMLR = plot_utils.Plot_Manifold_Learning_Result(RESULTS_DIR, PMLR_n_img_x, PMLR_n_img_y, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PMLR_resize_factor, PMLR_z_range)

        x_PMLR = test_data[0:PMLR_n_samples, :]
        id_PMLR = test_labels[0:PMLR_n_samples, :]

        if ADD_NOISE:
            x_PMLR = x_PMLR * np.random.randint(2, size=x_PMLR.shape)
            x_PMLR += np.random.randint(2, size=x_PMLR.shape)

    # Plot for analogy result
    if PARR and dim_z == 2:
        PARR = plot_utils.Plot_Analogical_Reasoning_Result(RESULTS_DIR, dim_z, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST, PARR_resize_factor, PARR_z_range)

    if (PMLR or PARR) and dim_z == 2:
        decoded = vae.decoder(z_in, fack_id_in, dim_img, n_hidden)

    # train
    total_batch = int(n_samples / batch_size)
    min_tot_loss = 1e99

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})

        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_total_data)
            train_data_ = train_total_data[:, :-mnist_data.NUM_LABELS]
            train_labels_ = train_total_data[:, -mnist_data.NUM_LABELS:]

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % (n_samples)
                batch_xs_input = train_data_[offset:(offset + batch_size), :]
                batch_ys_input = train_labels_[offset:(offset + batch_size)]
                batch_xs_target = batch_xs_input

                # add salt & pepper noise
                if ADD_NOISE:
                    batch_xs_input = batch_xs_input * np.random.randint(2, size=batch_xs_input.shape)
                    batch_xs_input += np.random.randint(2, size=batch_xs_input.shape)

                _, tot_loss, loss_likelihood, loss_divergence = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, y: batch_ys_input, keep_prob : 0.9})

            # print cost every epoch
            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss, loss_likelihood, loss_divergence))

            # if minimum loss is updated or final epoch, plot results
            if min_tot_loss > tot_loss or epoch+1 == n_epochs:
                min_tot_loss = tot_loss
                # Plot for reproduce performance
                if PRR:
                    x__PRR = sess.run(x_, feed_dict={x_hat: x_PRR, y: id_PRR, keep_prob : 1})
                    x__PRR_img = x__PRR.reshape(PRR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                    PRR.save_images(x__PRR_img, name="/PRR_epoch_%02d" %(epoch) + ".jpg")

                # Plot for manifold learning result
                if PMLR and dim_z == 2:

                    target_labels = [epoch % 10]

                    # If it is the final epoch, plot results for all labels
                    if epoch+1 == n_epochs:
                        target_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

                    for label in target_labels:
                        fake_id_PMLR = np.zeros(shape=[PMLR.z.shape[0],mnist_data.NUM_LABELS])
                        fake_id_PMLR[:,label] = 1.0
                        x__PMLR = sess.run(decoded, feed_dict={z_in: PMLR.z, fack_id_in:  fake_id_PMLR, keep_prob : 1})
                        x__PMLR_img = x__PMLR.reshape(PMLR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                        PMLR.save_images(x__PMLR_img, name="/PMLR_epoch_%02d_%02d" % (epoch, label) + ".jpg")

                    # plot distribution of labeled images
                    z_PMLR = sess.run(z, feed_dict={x_hat: x_PMLR, y: id_PMLR, keep_prob : 1})
                    PMLR.save_scattered_image(z_PMLR,id_PMLR, name="/PMLR_map_epoch_%02d" % (epoch) + ".jpg")

                # Plot for analogical reasoning result
                if epoch + 1 == n_epochs and PARR and dim_z == 2:
                    fake_id_PARR = np.zeros(shape=[PARR.z.shape[0], mnist_data.NUM_LABELS])
                    for i in range(PARR.z.shape[0]):
                        if i%11 == 0: # template
                            label = 3 #let's fix label for template as 3 for better style-comparison.
                        else:
                            label = (i % 11) - 1
                        fake_id_PARR[i, label] = 1.0
                    x__PARR = sess.run(decoded, feed_dict={z_in: PARR.z, fack_id_in: fake_id_PARR, keep_prob: 1})
                    x__PARR_img = x__PARR.reshape(PARR.n_tot_imgs, IMAGE_SIZE_MNIST, IMAGE_SIZE_MNIST)
                    PARR.save_images(x__PARR_img, name="/PARR.jpg")

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)