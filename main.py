import numpy as np
from timegan import timegan
from utils.data_utils import real_data_loading, sine_data_generation
import argparse
from utils.visualization_metrics import visualization
import time 

def main (args):

    ## Data loading
    if args.data_name in ['stock']:
        ori_data = real_data_loading(args.data_name, args.seq_len)
    elif args.data_name == 'sine':
        # Set number of samples and its dimensions
        no, dim = 10000, 5
        seq_len = args.seq_len
        ori_data = sine_data_generation(no, seq_len, dim)

        
    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()  
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['num_epochs'] = args.n_epochs
    parameters['batch_size'] = args.batch_size
    parameters['dest'] = args.dest
      
    generated_data = timegan(ori_data, parameters)   
    import pdb; pdb.set_trace() 
    return ori_data, generated_data



if __name__ == '__main__':  
  
    # Inputs for the main function
    date_ = time.strftime("%Y-%m-%d_%H%M")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_name', choices=['sine','stock'], default='sine', type=str, help='source dataset')
    parser.add_argument('--seq_len', default=24, type=int, help='sequence length')
    parser.add_argument('--hidden_dim', default=24,type=int, help='hidden state dimensions (should be optimized)')
    parser.add_argument('--num_layer', default=3,type=int, help='number of layers (should be optimized)')
    parser.add_argument('--n_epochs', default=50, type=int, help='Training epochs (should be optimized)',)
    parser.add_argument('--batch_size', default=128, type=int, help='the number of samples in mini-batch (should be optimized)')  
    parser.add_argument('--dest', default='exp_', type=str, help='experiment dir')
    args = parser.parse_args() 

    # Calls main function  
    ori_data, generated_data = main(args)