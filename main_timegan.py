"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
from readinData import getData, splitData
from metrics.fidelitytest import fidelity
import os


def complete():
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    os.system('say "your program has finished"')


def main(args):
    """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """

    ## Data loading
    if args.data_name in ['stock', 'energy']:
        ori_data = real_data_loading(args.data_name, args.seq_len)
        print(np.array(ori_data).shape)
    elif args.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        ori_data = sine_data_generation(no, args.seq_len, dim)
        # print(np.array(ori_data).shape)
    elif args.data_name == 'fred':
        d = [1, 2]
        ma = [3, 6]
        data = getData(d=d, ma=ma)
        ori_data, y_train_1, _1, _2, _3, _4 = splitData(data, var='UNRATE', inval=36, outval=1, holdout=False,
                                                        val=False,
                                                        perc=.90)

    print(args.data_name + ' dataset is ready.')

    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size

    try:
        os.mkdir("visualizations/RUN{}_{}_{}".format(parameters['hidden_dim'],
                                                     parameters['num_layer'],
                                                     parameters['iterations']))
        os.mkdir("visualizations/RUN{}_{}_{}/datasets".format(parameters['hidden_dim'],
                                                              parameters['num_layer'],
                                                              parameters['iterations']))
        os.mkdir("visualizations/RUN{}_{}_{}/PCA".format(parameters['hidden_dim'],
                                                         parameters['num_layer'],
                                                         parameters['iterations']))
        os.mkdir("visualizations/RUN{}_{}_{}/tSNE".format(parameters['hidden_dim'],
                                                          parameters['num_layer'],
                                                          parameters['iterations']))
    except OSError:
        print("Creation of the directory %s failed")
    else:
        print("Successfully created the directory %s")

    generated_data, generated_data2 = timegan(ori_data, parameters)
    print('Finish Synthetic Data Generation')

    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data, generated_data)
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(ori_data, generated_data)
        predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(ori_data, generated_data, 'pca')
    visualization(ori_data, generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)

    # metric_results['fidelity'], _ = fidelity(ori_data, generated_data)
    np.savez('visualizations/RUN{}_{}_{}/timegan_results_final.npz',
             ori=ori_data, gen=generated_data, gen_long=generated_data2,
             metrics=metrics)

    return ori_data, generated_data, generated_data2, metric_results


# if __name__ == '__main__':
# Inputs for the main function
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_name',
    choices=['sine', 'stock', 'energy', 'fred'],
    default='fred',
    type=str)
parser.add_argument(
    '--seq_len',
    help='sequence length',
    default=36,
    type=int)
parser.add_argument(
    '--module',
    choices=['gru', 'lstm', 'lstmLN'],
    default='lstm',
    type=str)
parser.add_argument(
    '--hidden_dim',
    help='hidden state dimensions (should be optimized)',
    default=25,
    type=int)
parser.add_argument(
    '--num_layer',
    help='number of layers (should be optimized)',
    default=3,
    type=int)
parser.add_argument(
    '--iteration',
    help='Training iterations (should be optimized)',
    default=10000,
    type=int)
parser.add_argument(
    '--batch_size',
    help='the number of samples in mini-batch (should be optimized)',
    default=128,
    type=int)
parser.add_argument(
    '--metric_iteration',
    help='iterations of the metric computation',
    default=10,
    type=int)

args = parser.parse_args()

# Calls main function
ori_data, generated_data, generated_data2, metrics = main(args)

complete()
