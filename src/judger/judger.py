import json


class Judger:
    # Initialize Judger, with the path of accusation list and law articles list
    def __init__(self, accusation_path, law_path):
        self.accu_dic = {}

        f = open(accusation_path, 'r')
        self.task1_cnt = 0
        for line in f:
            self.task1_cnt += 1
            self.accu_dic[line[:-1]] = self.task1_cnt

        self.law_dic = {}
        f = open(law_path, 'r')
        self.task2_cnt = 0
        for line in f:
            self.task2_cnt += 1
            self.law_dic[int(line[:-1])] = self.task2_cnt

    # Gen new results according to the truth and users output
    def gen_new_result(self, result, truth, label):
        s1 = set(label['accusation'])
        s2 = set()
        for name in truth['accusation']:
            s2.add(self.accu_dic[name.replace('[', '').replace(']', '')])

        for a in range(0, self.task1_cnt):
            in1 = (a + 1) in s1
            in2 = (a + 1) in s2
            if in1:
                if in2:
                    result[0][a]['TP'] += 1
                else:
                    result[0][a]['FP'] += 1
            else:
                if in2:
                    result[0][a]['FN'] += 1
                else:
                    result[0][a]['TN'] += 1

        s1 = set(label['articles'])
        s2 = set()
        for name in truth['relevant_articles']:
            s2.add(self.law_dic[name])

        for a in range(0, self.task2_cnt):
            in1 = (a + 1) in s1
            in2 = (a + 1) in s2
            if in1:
                if in2:
                    result[1][a]['TP'] += 1
                else:
                    result[1][a]['FP'] += 1
            else:
                if in2:
                    result[1][a]['FN'] += 1
                else:
                    result[1][a]['TN'] += 1

        return result

    # Calculate precision, recall and f1 value
    # According to https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
    @staticmethod
    def get_value(res):
        if res['TP'] == 0:
            if res['FP'] == 0 and res['FN'] == 0:
                precision = 1.0
                recall = 1.0
                f1 = 1.0
            else:
                precision = 0.0
                recall = 0.0
                f1 = 0.0
        else:
            precision = 1.0 * res['TP'] / (res['TP'] + res['FP'])
            recall = 1.0 * res['TP'] / (res['TP'] + res['FN'])
            f1 = 2 * precision * recall / (precision + recall)

        return precision, recall, f1

    def test(self, truth_file, output_file):
        cnt = 0
        result = [[], []]
        for a in range(0, self.task1_cnt):
            result[0].append({'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
        for a in range(0, self.task2_cnt):
            result[1].append({'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})

        truth_f = open(truth_file, 'r', encoding='utf-8')
        output_f = open(output_file, 'r', encoding='utf-8')

        for line in truth_f:
            ground_truth = json.loads(line)['meta']
            user_output = json.loads(output_f.readline())

            cnt += 1
            result = self.gen_new_result(result, ground_truth, user_output)

        return result

    def calc_f1(self, result):
        new_result = []
        sum_f = 0
        y = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for res in result:
            p, r, f = self.get_value(res)
            sum_f += f
            for z in res.keys():
                y[z] += res[z]

            res['F1'] = f
            new_result.append(res)

        _, _, micro_f = self.get_value(y)
        macro_f = sum_f / len(result)

        return micro_f, macro_f, new_result
