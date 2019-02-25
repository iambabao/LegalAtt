import os
import tensorflow as tf

from src import config
from src.judger import Judger

print('Current model: ', config.CURRENT_MODEL)
if config.CURRENT_MODEL == 'fasttext':
    from src.predict.predict_fasttext import predict
elif config.CURRENT_MODEL == 'bilstm':
    from src.predict.predict_bilstm import predict
elif config.CURRENT_MODEL == 'bilstm_att':
    from src.predict.predict_bilstm_att import predict
elif config.CURRENT_MODEL == 'bigru':
    from src.predict.predict_bigru import predict
elif config.CURRENT_MODEL == 'bigru_att':
    from src.predict.predict_bigru_att import predict
elif config.CURRENT_MODEL == 'han':
    from src.predict.predict_han import predict
elif config.CURRENT_MODEL == 'cnn':
    from src.predict.predict_cnn import predict
elif config.CURRENT_MODEL == 'dpcnn':
    from src.predict.predict_dpcnn import predict
elif config.CURRENT_MODEL == 'transformer':
    from src.predict.predict_transformer import predict
elif config.CURRENT_MODEL == 'topjudge':
    from src.predict.predict_topjudge import predict
elif config.CURRENT_MODEL == 'fact_law':
    from src.predict.predict_fact_law import predict
elif config.CURRENT_MODEL == 'law_att':
    from src.predict.predict_law_att import predict
else:
    print('No model named: ', config.CURRENT_MODEL)
    exit()

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID
config_proto = tf.ConfigProto(allow_soft_placement=True)  # 创建配置，允许将无法放入GPU的操作放在CUP上执行
config_proto.gpu_options.allow_growth = True  # 运行时动态增加内存使用量
judger = Judger(config.ACCU_DICT, config.LAW_DICT)
predict(judger, config_proto)
