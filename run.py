from argparse import Namespace
import co2_dataset
import os
import time

                                         
                                                                                                                                                       
# Settings
data_path         = 'CO2/monthly_in_situ_co2_mlo.csv'
save_path         = 'reg_params/params3'
epochs            = 5000
minibatch_size    = 100
mc_samples        = 20
optimizer         = 'adam'
learning_rate     = 1e-2
momentum          = 0.9
l2_weight         = 1e-6
drop_p            = 0.1
tau_rc            = 0.07
tau_ref           = 0.0005
amplitude         = 0.01
train             = True
continue_training = False
spiking           = False
plot              = True
comment           = 'test run'

args = Namespace(data_path=data_path, epochs=epochs, minibatch_size=minibatch_size,
                 optimizer=optimizer, learning_rate=learning_rate, l2_weight=l2_weight, momentum=momentum,
                 mc_samples=mc_samples, tau_ref=tau_ref, tau_rc=tau_rc, train=train, continue_training=continue_training,
                 save_path=save_path, amplitude=amplitude, drop_p=drop_p, spiking=spiking, plot=plot)

print('########################')
print(comment) # a comment that will be printed in the log file
print(args)    # print all args in the log file so we know what we were running
print('########################')


start = time.time()
loss = co2_dataset.main(args)
print("The training took {:.1f} minutes with a loss of {:.3f}".format((time.time()-start)/60,loss)) # measure time
