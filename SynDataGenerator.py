##This script generates sequences based on several differen models,
# and saves it in the project data diectory as well as MATLAB directory

import utils
import numpy as np
import scipy.io

generation_type = 'recurrent_unique'

if generation_type == 'recurrent':

    W,Hx,Hh,beta,st, entropy, x_train, x_valid, x_test = utils.generate_syn(vocab_size=1000,
                                                                            state_size=100,
                                                                            train_seq_len=500000,
                                                                            test_seq_len=40000,
                                                                            valid_seq_len=40000)

    np.savez('../data/syn/syn_data',W=W,Hx=Hx,Hh=Hh, beta=beta,
             x_train=x_train,x_test=x_test,x_valid=x_valid,entropy=entropy)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data.mat', mdict={'W':W,
                                                                          'Hx':Hx,
                                                                          'Hh':Hh,
                                                                          'beta':beta,
                                                                          'x_train': x_train,
                                                                          'x_test':x_test,
                                                                          'x_valid':x_valid})

if generation_type == 'recurrent_unique':

    W,Hx,Hh,beta,st, entropy, x_train, x_valid, x_test = utils.generate_syn_unique(vocab_size=100,
                                                                            state_size=100,
                                                                            train_seq_len=500000,
                                                                            test_seq_len=40000,
                                                                            valid_seq_len=40000)

    np.savez('../data/syn/syn_data_unique',W=W,Hx=Hx,Hh=Hh, beta=beta,
             x_train=x_train,x_test=x_test,x_valid=x_valid,entropy=entropy)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_unique.mat', mdict={'W':W,
                                                                          'Hx':Hx,
                                                                          'Hh':Hh,
                                                                          'beta':beta,
                                                                          'x_train': x_train,
                                                                          'x_test':x_test,
                                                                          'x_valid':x_valid})


if generation_type == 'recurrent_closed_hmm':

    W,Hx,Hh,beta,st, entropy, x_train, x_valid, x_test = utils.generate_syn_closed_hmm(vocab_size=1000,
                                                                            state_size=100,
                                                                            train_seq_len=500000,
                                                                            test_seq_len=40000,
                                                                            valid_seq_len=40000)

    np.savez('../data/syn/syn_data_closed_hmm',W=W,Hx=Hx,Hh=Hh, beta=beta,
             x_train=x_train,x_test=x_test,x_valid=x_valid,entropy=entropy)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_closed_hmm.mat', mdict={'W':W,
                                                                          'Hx':Hx,
                                                                          'Hh':Hh,
                                                                          'beta':beta,
                                                                          'x_train': x_train,
                                                                          'x_test':x_test,
                                                                          'x_valid':x_valid})


elif generation_type == 'recurrent_nonlinear':

    W,Hx,Hh,beta,st, entropy, x_train, x_valid, x_test = utils.generate_syn_nonlinear(vocab_size=1000,
                                                                            state_size=100,
                                                                            train_seq_len=500000,
                                                                            test_seq_len=40000,
                                                                            valid_seq_len=40000,
                                                                            percentile=80)

    np.savez('../data/syn/syn_data_nonlinear_80p',W=W,Hx=Hx,Hh=Hh, beta=beta,
             x_train=x_train,x_test=x_test,x_valid=x_valid,entropy=entropy)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_nonlinear_80p.mat', mdict={'W':W,
                                                                          'Hx':Hx,
                                                                          'Hh':Hh,
                                                                          'beta':beta,
                                                                          'x_train': x_train,
                                                                          'x_test':x_test,
                                                                          'x_valid':x_valid})

elif generation_type == 'recurrent_toy':
    W, Hx, Hh, beta, st, entropy, x_train, x_valid, x_test = utils.generate_syn(vocab_size=10,
                                                                                state_size=4,
                                                                                train_seq_len=500000,
                                                                                test_seq_len=40000,
                                                                                valid_seq_len=40000)

    np.savez('../data/syn/syn_data_toy', W=W, Hx=Hx, Hh=Hh, beta=beta,
             x_train=x_train, x_test=x_test, x_valid=x_valid, entropy=entropy)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_toy.mat', mdict={'W': W,
                                                                          'Hx': Hx,
                                                                          'Hh': Hh,
                                                                          'beta': beta,
                                                                          'x_train': x_train,
                                                                          'x_test': x_test,
                                                                          'x_valid': x_valid})

elif generation_type == 'low-rank':
    W,H,st, entropy, x_train, x_valid, x_test = utils.generate_syn_lr(vocab_size=1000,
                                                                      state_size=100,
                                                                      train_seq_len=500000,
                                                                      test_seq_len=40000,
                                                                      valid_seq_len=40000)

    np.savez('../data/syn/syn_data_lr',W=W,H=H,st=st, entropy=entropy,
             x_train=x_train,x_test=x_test,x_valid=x_valid)

    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_lr.mat', mdict={'W':W,
                                                                          'H':H,
                                                                            'st':st,
                                                                            'entropy':entropy,
                                                                          'x_train': x_train,
                                                                          'x_test':x_test,
                                                                          'x_valid':x_valid})


elif generation_type == 'low-rank-bigrams':
    W, H, entropy, st, x_train, x_valid, x_test = utils.generate_syn_non_recur_mixed(vocab_size=1000,
                                                                                     state_size=100,
                                                                                     train_seq_len=500000,
                                                                                     test_seq_len=40000,
                                                                                     valid_seq_len=40000)
    np.savez('../data/syn/syn_data_non_recur_mixed_k1000', W=W, H=H, entropy=entropy, st=st,
             x_train=x_train, x_test=x_test, x_valid=x_valid)
    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_non_recur_mixed_k1000.mat', mdict={'W': W,
                                                                                    'H': H,
                                                                                    'ent':entropy,
                                                                                    'st' : st,
                                                                                    'syn_train': x_train,
                                                                                    'syn_test': x_test,
                                                                                    'syn_valid': x_valid})

elif generation_type == 'hmm-nonlinear':

    W,H,st,entropy,x_train, x_valid, x_test = utils.generate_syn_hmm_nonlinear(vocab_size=1000,state_size=100,
                                                      train_seq_len=500000, test_seq_len=40000,
                                                      valid_seq_len=40000,percentile=90)
    np.savez('../data/syn/syn_data_hmm_nonlinear',W=W, H=H,st=st,entropy=entropy,
             x_train=x_train,x_test=x_test,x_valid=x_valid)
    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_hmm_nonlinear.mat', mdict={'W':W,
                                                                                    'H':H,
                                                                                    'st':st,
                                                                                    'entropy':entropy,
                                                                                    'x_train': x_train,
                                                                                    'x_test':x_test,
                                                                                    'x_valid':x_valid})

elif generation_type == 'hmm':

    W,H,st,entropy,x_train, x_valid, x_test = utils.generate_syn_hmm(vocab_size=1000,state_size=100,
                                                      train_seq_len=500000, test_seq_len=40000,
                                                      valid_seq_len=40000)
    np.savez('../data/syn/syn_data_hmm',W=W, H=H,st=st,entropy=entropy,
             x_train=x_train,x_test=x_test,x_valid=x_valid)
    scipy.io.savemat('/Users/Moein/Documents/MATLAB/syn_data_hmm.mat', mdict={'W':W,
                                                                                    'H':H,
                                                                                    'st':st,
                                                                                    'entropy':entropy,
                                                                                    'x_train': x_train,
                                                                                    'x_test':x_test,
                                                                                    'x_valid':x_valid})