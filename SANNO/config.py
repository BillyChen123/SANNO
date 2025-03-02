import argparse

def parser_add_main_args():
    parser = argparse.ArgumentParser(description='Code for STOT',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--gpu_index', type=str, default='1', help='')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--name', type=str, default='test')

    parser.add_argument('--dataset', type=str, default='Hubmap_CL', help='dataset')
    parser.add_argument('--train_dataset', type=str, default='',help='path to train_h5ad')
    parser.add_argument('--test_dataset',type=str, default='',help='path to test_h5ad')
    parser.add_argument('--log', type=str, default='', help='path to log')

    parser.add_argument('--type', type=str, default='st2st', help='annotation type')
    parser.add_argument('--class_balance', action='store_true', help='use class balance', default=True)
    parser.add_argument('--num_workers', type=int, help='num workers', default=4)
    parser.add_argument('--batch_size', type=int, help='batch size', default=36)
    parser.add_argument('--num_layers', type=int, help='num_layer', default=3)
    parser.add_argument('--tau', type=float, default=0.25, help='temperature for gumbel softmax')
    
    parser.add_argument('--K', type=int, help='K', default=50)
    parser.add_argument('--gamma', type=float, help='gamma', default=0.5)
    parser.add_argument('--mu', type=float, help='mu', default=0.7)
    parser.add_argument('--temp', type=float, help='temp', default=0.1)
    parser.add_argument('--lam_PCD', type=float, help='lam_PCD', default=0.1)
    parser.add_argument('--lam_CCD', type=float, help='lam_CCD', default=0.1)
    parser.add_argument('--lam_ent', type=float, help='lam_ent', default=0.1)

    parser.add_argument('--lam_local', type=float, help='lam_local', default=0.5)
    parser.add_argument('--lam_global', type=float, help='lam_global', default=0.5)
    parser.add_argument('--lam_pe', type=float, help='lam_pe', default=1)
    parser.add_argument('--lam_ne', type=float, help='lam_ne', default=2)
    parser.add_argument('--lam_link', type=float, help='lam_link', default=1)

    parser.add_argument('--MQ_size', type=int, help='MQ_size', default=5000)

    parser.add_argument('--min_step', type=int, help='min_step', default=5000)
    parser.add_argument('--lr', type=float, help='lr', default=0.01)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=0.0005)
    parser.add_argument('--sgd_momentum', type=float, help='sgd_momentum', default=0.9)

    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--root_dir', type=str, default="log/MLP_1020_baseline_B004_reg003_only_ce")

    parser.add_argument('--feat_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()
    return args