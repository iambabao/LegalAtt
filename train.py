import os
import tensorflow as tf

from src import config
from src.judger import Judger

print('Current model: ', config.CURRENT_MODEL)
if config.CURRENT_MODEL == 'fasttext':
    from src.train.train_fasttext import train
elif config.CURRENT_MODEL == 'bilstm':
    from src.train.train_bilstm import train
elif config.CURRENT_MODEL == 'bilstm_att':
    from src.train.train_bilstm_att import train
elif config.CURRENT_MODEL == 'bigru':
    from src.train.train_bigru import train
elif config.CURRENT_MODEL == 'bigru_att':
    from src.train.train_bigru_att import train
elif config.CURRENT_MODEL == 'han':
    from src.train.train_han import train
elif config.CURRENT_MODEL == 'cnn':
    from src.train.train_cnn import train
elif config.CURRENT_MODEL == 'dpcnn':
    from src.train.train_dpcnn import train
elif config.CURRENT_MODEL == 'transformer':
    from src.train.train_transformer import train
elif config.CURRENT_MODEL == 'topjudge':
    from src.train.train_topjudge import train
elif config.CURRENT_MODEL == 'fact_law':
    from src.train.train_fact_law import train
else:
    exit(1)

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
config_proto = tf.ConfigProto(allow_soft_placement=True)  # 创建配置，允许将无法放入GPU的操作放在CUP上执行
config_proto.gpu_options.allow_growth = True  # 运行时动态增加内存使用量
judger = Judger(config.ACCU_DICT, config.LAW_DICT)
train(judger, config_proto)
