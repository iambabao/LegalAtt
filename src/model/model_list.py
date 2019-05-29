from .cnn import CNN
from .dpcnn import DPCNN
from .lstm import LSTM
from .gru import GRU
from .topjudge import TopJudge
from .legal_att import LegalAtt

model_list = {
    'cnn': CNN,
    'dpcnn': DPCNN,
    'lstm': LSTM,
    'gru': GRU,
    'topjudge': TopJudge,
    'legal_att': LegalAtt,
}


def get_model(config, embedding_matrix, is_training):
    assert config.current_model in model_list

    model = model_list[config.current_model](config, embedding_matrix=embedding_matrix, is_training=is_training)

    return model
