import os
import json

from src import util


class DataReader:
    def __init__(self, config):
        self.config = config
        self.stop_word = self.load_stop_word()

    def load_stop_word(self):
        stop_word = []
        with open(self.config.stop_word, 'r', encoding='utf-8') as fin:
            for line in fin:
                stop_word.append(line.strip())
        return stop_word

    def remove_stop_word(self, word_list):
        # return [word for word in word_list if word not in self.stop_word]
        return word_list

    def read_data(self, data_file, word_2_id, accu_2_id, art_2_id):
        fact = []
        fact_len = []
        accu = []
        relevant_art = []
        impr = []

        with open(data_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                item = json.loads(line)

                temp = self.remove_stop_word(item['fact'].strip().split())
                temp = util.convert_list(temp, word_2_id, self.config.pad_id, self.config.unk_id)
                temp = temp[:self.config.sequence_len]
                fact.append(temp)

                fact_len.append(len(temp))

                temp = [0] * self.config.accu_num
                t = [accu_2_id[v] for v in item['meta']['accusation']]
                for v in t:
                    temp[v] = 1
                accu.append(temp)

                temp = [0] * self.config.art_num
                t = [str(v) for v in item['meta']['relevant_articles']]
                t = [art_2_id[v] for v in t]
                for v in t:
                    temp[v] = 1
                relevant_art.append(temp)

                temp = [0] * self.config.impr_num
                temp[util.impr_2_id(item['meta']['term_of_imprisonment'])] = 1
                impr.append(temp)

        return fact, fact_len, accu, relevant_art, impr

    def read_train_data(self, word_2_id, accu_2_id, art_2_id):
        return self.read_data(self.config.train_data, word_2_id, accu_2_id, art_2_id)

    def read_valid_data(self, word_2_id, accu_2_id, art_2_id):
        return self.read_data(self.config.valid_data, word_2_id, accu_2_id, art_2_id)

    def read_test_data(self, word_2_id, accu_2_id, art_2_id):
        return self.read_data(self.config.test_data, word_2_id, accu_2_id, art_2_id)

    def read_article(self, art_list, word_2_id):
        art = []
        art_len = []

        for art_name in art_list:
            file_name = os.path.join(self.config.criminal_law_dir, art_name + '.txt')
            with open(file_name, 'r', encoding='utf-8') as fin:
                content = fin.readline()
                content = util.cut_text(content)
                content = util.convert_list(content, word_2_id, self.config.pad_id, self.config.unk_id)
                content = content[:self.config.sequence_len]

                art.append(util.pad_list(content, self.config.pad_id, self.config.sequence_len))

                art_len.append(len(content))

        return art, art_len

    def convert_data(self, data, word_2_id):
        fact = []
        fact_len = []

        temp = self.remove_stop_word(util.refine_text(data))
        temp = util.convert_list(temp, word_2_id, self.config.pad_id, self.config.unk_id)
        temp = temp[:self.config.sequence_len]

        fact.append(temp)

        fact_len.append(len(temp))

        return fact, fact_len
