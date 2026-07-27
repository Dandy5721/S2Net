import argparse

parser = argparse.ArgumentParser(description='PyTorch IMDB Training')

parser.add_argument('--epochs', default=500, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--dataset', default='HCPYA', type=str, metavar='N',
					help='the name of dataset')
parser.add_argument('--batch_size', default=16, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--gpu', default='0', type=int, metavar='N',
					help='cuda')
parser.add_argument('--ablation', type=str, default='full', 
                    choices=['full', 'no_coupling', 'no_mask', 'no_smooth', 'raw_feature'],
                    help="""
                    full: 完整模型 (Standard)
                    no_coupling: 禁用 SC 结构连接 (Kuramoto 退化为独立振荡器)
                    no_mask: 禁用基于相位的动态 Mask (Gate) 机制
                    no_smooth: 禁用输出端的时序平滑 (Conv1d)
                    raw_feature: 完全移除 Kuramoto 层，直接使用 Linear 映射特征
                    """)
parser.add_argument('--hidden', default='64', type=int, metavar='N',
					help='cuda')
parser.add_argument('--k', default='1.0', type=float, metavar='N',
					help='cuda')
parser.add_argument('--ck', type=float, default=1.0, help='Coupling strength K')
parser.add_argument('--tau', type=int, default=2, help='Phase delay steps tau')
parser.add_argument('--osc_dim', type=int, default=4, help='Oscillator dimension D')
parser.add_argument('--dt', default='1.0', type=float, metavar='N',
					help='cuda')
parser.add_argument('-b', '--batch-size', default=16, type=int,metavar='N',
					help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--algo', default='SRNN', type=str, metavar='N',
					help='algorithmn for learning')
parser.add_argument('--thresh', default=0.5, type=float, metavar='N',
					help='threshold of the neuron model')
parser.add_argument('--lens', default=0.5, type=float, metavar='N',
					help='lens of surrogate function')
parser.add_argument('--decay', default=0.5, type=float, metavar='N',
					help='decay of the neuron model')
parser.add_argument('--in_size', default=4, type=int, metavar='N',
					help='model input size')
parser.add_argument('--out_size', default=9, type=int, metavar='N',
					help='model output size')
parser.add_argument('--branch', default=1, type=int, metavar='N',
					help='branch of neuron model')
parser.add_argument('--low_n', default=0, type=int, metavar='N',
					help='branch of neuron model')
parser.add_argument('--high_n', default=4, type=int, metavar='N',
					help='branch of neuron model')

parser.add_argument('--phase_max', nargs='*', default=[0.0, 0.0], type=float, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--cycle_min', nargs='*', default=[10, 10], type=int, metavar='N',
					help='minimum repeating cycle')
parser.add_argument('--cycle_max', nargs='*', default=[50, 50], type=int, metavar='N',
					help='maximum repeating cycle')
parser.add_argument('--duty_cycle_min', nargs='*', default=[0.95, 0.95], type=float, metavar='L',
					help='min duty cycle for each layer')
parser.add_argument('--duty_cycle_max', nargs='*', default=[1.0, 1.0], type=float, metavar='L',
					help='max duty cycle for each layer')
					
parser.add_argument('--fc', nargs= '+', default=[1024], type=int, metavar='N',
					help='model architecture')
parser.add_argument('--num_steps', default=1301, type=int, metavar='N',
					help='time steps')
parser.add_argument('--chkp_path', default='', type=str, metavar='PATH',
					help='path to save the training model (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
					help='path to save the training record (default: none)')

args = parser.parse_args()