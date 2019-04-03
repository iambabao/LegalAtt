import os
import argparse
import tensorflow as tf

from src.config import Config
from src.judger import Judger

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--embedding_trainable', action='store_true', default=False)
parser.add_argument('--use_batch_norm', action='store_true', default=False)
args = parser.parse_args()

current_model = args.model
num_epoch = args.num_epoch
batch_size = args.batch_size
optimizer = args.optimizer
lr = args.lr
embedding_trainable = args.embedding_trainable
use_batch_norm = args.use_batch_norm
config = Config('./', current_model,
                num_epoch=num_epoch, batch_size=batch_size, optimizer=optimizer, lr=lr,
                embedding_trainable=embedding_trainable, use_batch_norm=use_batch_norm)

print('Current model: ', config.current_model)
if config.current_model == 'fasttext':
    from src.train.train_fasttext import train
elif config.current_model == 'bilstm':
    from src.train.train_bilstm import train
elif config.current_model == 'bigru':
    from src.train.train_bigru import train
elif config.current_model == 'han':
    from src.train.train_han import train
elif config.current_model == 'cnn':
    from src.train.train_cnn import train
elif config.current_model == 'dpcnn':
    from src.train.train_dpcnn import train
elif config.current_model == 'topjudge':
    from src.train.train_topjudge import train
elif config.current_model == 'fact_law':
    from src.train.train_fact_law import train
elif config.current_model == 'law_att':
    from src.train.train_law_att import train
else:
    print('No model named: ', config.current_model)
    exit()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_proto = tf.ConfigProto(allow_soft_placement=True)  # 创建配置，允许将无法放入GPU的操作放在CUP上执行
config_proto.gpu_options.allow_growth = True  # 运行时动态增加内存使用量
judger = Judger(config.accu_dict, config.law_dict)

train(config, judger, config_proto)
