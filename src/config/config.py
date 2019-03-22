import os


class Config:
    def __init__(self, root_dir, current_model, top_k=5, embedding_size=200, tfidf_size=5000,
                 hidden_size=100, att_size=100, kernel_size=(2, 3, 4, 5), filter_dim=50,
                 block_num=5, head_num=5, model_dim=200, fc_size_s=200, fc_size_m=500, fc_size_l=1000,
                 num_epoch=30, batch_size=16, sentence_len=500, sequence_len=100, document_len=20,
                 optimizer='Adam', lr=0.001, keep_prob=0.8, l2_rate=0.0, use_batch_norm=False):

        self.root_dir = root_dir

        # 存储临时文件
        self.temp_dir = os.path.join(self.root_dir, 'temp')

        # 存储数据文件
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.train_data = os.path.join(self.data_dir, 'data_train.json')
        self.valid_data = os.path.join(self.data_dir, 'data_valid.json')
        self.test_data = os.path.join(self.data_dir, 'data_test.json')
        self.word_count = os.path.join(self.data_dir, 'word_count.json')
        self.word_dict = os.path.join(self.data_dir, 'word_dict.json')
        self.accu_dict = os.path.join(self.data_dir, 'accu.txt')
        self.law_dict = os.path.join(self.data_dir, 'law.txt')

        # 存储法律文本
        self.law_kb_dir = os.path.join(self.data_dir, 'law_kb')
        
        # 存储词向量文件
        self.embedding_dir = os.path.join(self.data_dir, 'embedding')
        self.plain_text = os.path.join(self.embedding_dir, 'plain_text.txt')
        self.word2vec_model = os.path.join(self.embedding_dir, 'word2vec.model')
        self.tfidf_model = os.path.join(self.embedding_dir, 'tfidf.model')

        self.current_model = current_model        
        
        # 存储训练结果
        self.result_dir = os.path.join(self.root_dir, 'result', self.current_model)
        self.model_file = os.path.join(self.result_dir, 'model')
        self.valid_result = os.path.join(self.result_dir, 'valid_result')
        self.test_result = os.path.join(self.result_dir, 'test_result')

        self.pad = 'PAD'
        self.pad_id = 0
        self.unk = 'UNK'
        self.unk_id = 1
        self.num = 'NUM'
        self.num_id = 2
        self.vocab_size = 100000

        self.accu_num = 100
        self.article_num = 91
        self.imprisonment_num = 9
        self.task_threshold = (0.1, 0.2, 0.3, 0.4, 0.5)

        self.top_k = top_k
        self.embedding_size = embedding_size
        self.tfidf_size = tfidf_size

        # RNN
        self.hidden_size = hidden_size
        self.att_size = att_size

        # CNN
        self.kernel_size = kernel_size
        self.filter_dim = filter_dim

        # Transformer
        self.block_num = block_num
        self.head_num = head_num
        self.model_dim = model_dim

        # FC
        self.fc_size_s = fc_size_s
        self.fc_size_m = fc_size_m
        self.fc_size_l = fc_size_l

        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.sequence_len = sequence_len
        self.document_len = document_len
        
        self.optimizer = optimizer
        self.lr = lr
        self.keep_prob = keep_prob
        self.l2_rate = l2_rate
        self.use_batch_norm = use_batch_norm
