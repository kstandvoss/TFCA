# coding: utf-8

import nengo
import nengo_dl
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import argparse
import pdb


def main(args):

    co2_data = pd.read_csv(args.data_path, usecols=[0,4,5,6,7,8,9])
    co2_data.columns = ['Date', 'standard', 'season_adjust', 'smoothed', 'smoothed_season', 'standard_no_missing', 'season_no_missing']


    detrended = signal.detrend(co2_data['standard_no_missing'][200:600])
    detrended /= np.max(detrended)
    detrended *= 2

    #if args.plot:
    #    plt.plot(detrended)
    #    plt.axvline(x=300, c='black', lw='1')
    #    plt.ylim([-20,20])
    #    plt.xlim([0,500])

    # # Training setup

    # leaky integrate and fire parameters
    lif_params = {
        'tau_rc': args.tau_rc,
        'tau_ref': args.tau_ref,
        'amplitude': args.amplitude
    }


    # training parameters
    drop_p = args.drop_p
    minibatch_size = args.minibatch_size
    n_epochs = args.epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    l2_weight = args.l2_weight

    # lif parameters
    lif_neurons = nengo.LIF(**lif_params)

    # softlif parameters (lif parameters + sigma)
    softlif_neurons = nengo.RectifiedLinear()#nengo_dl.SoftLIFRate(**lif_params,sigma=0.002)

    # ensemble parameters
    ens_params = dict(max_rates=nengo.dists.Choice([100]), intercepts=nengo.dists.Choice([0]))


    def build_network(neuron_type, drop_p, l2_weight, n_units=1024, num_layers=4, output_size=1):
        with nengo.Network() as net:
            
            use_dropout = False
            if drop_p:
                use_dropout = True

            #net.config[nengo.Connection].synapse = None
            #nengo_dl.configure_settings(trainable=False)
            
            # input node
            inp = nengo.Node([0])
            
            shape_in = 1
            x = inp
            
            # the regularizer is a function, so why not reuse it
            reg = tf.contrib.layers.l2_regularizer(l2_weight)
            
            class DenseLayer(object):
                i=0
                def pre_build(self, shape_in, shape_out):
                    self.W = tf.get_variable(
                        "weights" + str(DenseLayer.i), shape=(shape_in[1], shape_out[1]),
                        regularizer=reg)
                    self.B = tf.get_variable(
                        "biases" + str(DenseLayer.i), shape=(1, shape_out[1]), regularizer=reg)
                    DenseLayer.i+=1

                def __call__(self, t, x):
                    pdb.set_trace()
                    return x @ self.W + self.B

            
            for n in range(num_layers):
                # add a fully connected layer
                a = nengo_dl.TensorNode(DenseLayer, size_in=shape_in, size_out=n_units)
                nengo.Connection(x, a)
                
                shape_in = n_units
                x = a
                
                # apply an activation function
                x = nengo_dl.tensor_layer(x, neuron_type, **ens_params)

                # add a dropout layer
                x = nengo_dl.tensor_layer(x, tf.layers.dropout, rate=drop_p, training=use_dropout)
                
            
            
            # add an output layer
            a = nengo_dl.TensorNode(DenseLayer, size_in=shape_in, size_out=output_size)
            nengo.Connection(x, a)

            
        return net, inp, a


    do_train = args.train
    continue_training = args.continue_training

    param_path = args.save_path


    trainset_size = len(detrended)

    x = np.linspace(-2,2,trainset_size)
    y = detrended


    # # training on continuous soft leaky integrate and fire neurons

    # construct the network
    net, inp, out = build_network(softlif_neurons, drop_p, l2_weight)
    with net:
        in_p = nengo.Probe(inp, 'output')
        out_p = nengo.Probe(out, 'output')
        
    """
    # define training set etc.
    """
    #pdb.set_trace()
    #train_x = {inp: x.reshape((minibatch_size, trainset_size // minibatch_size))[..., None]}
    #train_y = {out_p: y.reshape((minibatch_size, trainset_size // minibatch_size))[..., None]}
    target = x[:,None,None]
    train_x = {inp: target[:300]}
    train_y = {out_p: y[:300,None,None]}
    test_x = {inp: target[300:]}
    test_y = {out_p: y[300:,None,None]}

    # construct the simulator
    with nengo_dl.Simulator(net, minibatch_size=minibatch_size, tensorboard='./tensorboard') as sim:
        #, tensorboard='./tensorboard')
        
        # define the loss function (We need to do this in the
        # context of the simulator because it changes the
        # tensorflow default graph to the nengo network.
        # That is, tf.get_collection won't work otherwise.)
        def mean_squared_error_L2_regularized(y, t):

            if not y.shape.as_list() == t.shape.as_list():
                raise ValueError("Output shape", y.shape, "differs from target shape", t.shape)
            e = tf.reduce_mean((t - y)**2) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            return e

        with tf.name_scope('sum_weights'):
            first = 0
            for node in net.nodes:
                if type(node) == nengo_dl.tensor_node.TensorNode:
                    if 'Dense' in str(node.tensor_func):
                        if not first:
                            sum_weights = tf.linalg.norm(node.tensor_func.W)
                            first = 1
                        else:
                            sum_weights += tf.linalg.norm(node.tensor_func.W)
        weight_summary = tf.summary.scalar('sum_weights', sum_weights)        


        starter_learning_rate = args.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, sim.tensor_graph.training_step,
                                           1000, 0.96, staircase=True)
        

            # define optimiser  
        if args.optimizer=='rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif args.optimizer=='sgd':
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
        elif args.optimizer=='adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif args.optimizer=='adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        loss = 0
        # actual training loop
        if do_train:
            if continue_training:
                sim.load_params(path=param_path)

            loss = sim.loss(test_x, test_y, objective='mse')    
            print("error before training: ", loss)

            sim.train(train_x, train_y, opt, n_epochs=n_epochs, shuffle=True, objective={out_p:mean_squared_error_L2_regularized}, summaries=['loss', weight_summary])

            loss = sim.loss(test_x, test_y, objective='mse')
            print("error after training:", loss)

            sim.save_params(path=param_path)
        else:
            sim.load_params(path=param_path)

        #pdb.set_trace()
        T = args.mc_samples
        outputs = np.zeros((T,target.size))
        for t in range(T):
            for i in range(0,target.size,minibatch_size):
                sim.run_steps(1,input_feeds={inp: target[i:i+minibatch_size]})
                #outputs[t,i:i+minibatch_size] = np.squeeze(sim.data[out_p])
                sim.soft_reset(include_trainable=False, include_probes=False)
            outputs[t] = sim.data[out_p].transpose(1,0,2).reshape((len(target),))
            sim.soft_reset(include_trainable=False, include_probes=True)
            

        predictive_mean = np.mean(outputs, axis=0)
        predictive_variance = np.var(outputs, axis=0)   

        target = np.squeeze(target)

        if args.plot:
            plt.plot(target,predictive_mean,label='out')
            plt.fill_between(target, predictive_mean-2*np.sqrt(predictive_variance), predictive_mean+2*np.sqrt(predictive_variance),
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0, label='variance')
            plt.plot(target,detrended,label='target', color='blue',alpha=0.5)
            plt.axvline(x=x[300], c='black', lw='1')
            plt.ylim([-20,20])
            plt.legend(loc='upper right')



    if args.spiking:
        # # test on LIF neurons
        # timesteps
        #T = 50
        # MC dropout samples
        MC_drop = T

        # we want to see if spiking neural networks
        # need dropout at all, so we disable it
        net, inp, out = build_network(lif_neurons, drop_p=0, l2_weight=l2_weight)
        with net:
            in_p = nengo.Probe(inp)
            out_p = nengo.Probe(out)

            # start a new simulator
        # T is the amount of MC dropout samples
        sim = nengo_dl.Simulator(net, minibatch_size=len(target))#, unroll_simulation=10, tensorboard='./tensorboard')

        # load parameters
        sim.load_params(path=param_path)

        # copy the input for each MC dropout sample
        minibatched_target = np.tile(target[:, None], (1,T))[..., None]

        sim.soft_reset(include_trainable=False, include_probes=True)

        # run for T timesteps
        sim.run_steps(T, input_feeds={inp: minibatched_target})
            
        if args.plot:
            # plot
            plt.figure() 
            plt.scatter(sim.data[in_p].flatten(), sim.data[out_p].flatten(), c='r', s=1, label="output") 
            plt.plot()
            #plt.plot(target.flatten(), y(target).flatten(), label="target", linewidth=2.0)
            plt.legend(loc='upper left', bbox_to_anchor=(1.025, 1.025));
            plt.plot(detrended, label='train set')
            plt.axvline(x=x[300], c='black', lw='1')
            plt.ylim([-20,20])
            #plt.xlim([0,500]);


            # print(sim.data[out_p].shape)
            predictive_mean = np.squeeze(np.mean(sim.data[out_p][:, -MC_drop:, :], axis=1))
            predictive_variance = np.squeeze(np.var(sim.data[out_p][:, -MC_drop:, :], axis=1))


            plt.figure(figsize=(20, 10))
            plt.plot(target,predictive_mean,label='out')
            #plt.plot(target,spiking_outputs[:,-1],label='out')
            plt.fill_between(target, predictive_mean-predictive_variance, predictive_mean+predictive_variance,
                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', linewidth=0, label='variance')
            plt.plot(x, y, c='green', label='dataset')
            plt.scatter(x,y, color='black', s=9, label='train set')

            plt.axvline(x=300, c='black', lw='1')

            plt.legend(loc='upper left', bbox_to_anchor=(1.025, 1.025))
            plt.ylim([-20,20])
            plt.axvline(x=x[300], c='black', lw='1')
            #plt.xlim([0,500])
    
        sim.close()
    if args.plot:
        plt.show()

    return loss

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Train spiking neural network to perform variational inference on co2 dataset')
    parser.add_argument('data_path', action='store',
                        help='Path to data')
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-mb', action='store', dest='minibatch_size', type=int, default=25,
                        help='Size of training mini batches')
    parser.add_argument('-t', action='store', dest='mc_samples', type=int, default=20,
                        help='Number of MC forwardpasses and timesteps for spiking network')
    parser.add_argument('-o', '--optimizer', action='store', dest='optimizer', default='rmsprop', choices=('sgd', 'adadelta', 'adam', 'rmsprop'),
                        help='Optimization function')
    parser.add_argument('-r', '--learning_rate', action='store', dest='learning_rate', type=float,
                        help='Learning rate', default=1e-4)
    parser.add_argument('-m', '--momentum', action='store', dest='momentum', type=float,
                        help='Momentum', default=0.9)
    parser.add_argument('-l', '--l2_weight', action='store', dest='l2_weight', type=float,
                        help='Weight of l2 regularization', default=1e-6)     
    parser.add_argument('-d', '--dropout', action='store', dest='drop_p', type=float,
                        help='Dropout probability', default=0.1)    
    parser.add_argument('-rc', '--tau_rc', action='store', dest='tau_rc', type=float,
                        help='LIF parameter', default=0.07)  
    parser.add_argument('-ref', '--tau_ref', action='store', dest='tau_ref', type=float,
                        help='LIF parameter', default=0.0005)  
    parser.add_argument('-a', '--amplitude', action='store', dest='amplitude', type=float,
                        help='LIF parameter', default=0.05)                                                                                                                     
    parser.add_argument('--save_path', action='store', default='./reg_params/params')
    parser.add_argument('--train', action='store_true', dest='train', default=True,
                        help='Train new network, else load parameters')
    parser.add_argument('--continue_training', action='store_true', dest='continue_training', default=False,
                        help='Continue training from previous parameters')
    parser.add_argument('--plot', action='store_true', dest='plot', default=False,
                        help='Plot results')
    parser.add_argument('--spiking', action='store_true', dest='spiking', default=False,
                        help='Test spiking model')

    args = parser.parse_args()

    main(args)


