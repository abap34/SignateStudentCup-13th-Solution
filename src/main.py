import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from simpletransformers.classification import ClassificationModel
import pprint
# import seaborn as sns
import yaml
import glob
import os

languages = ['en', 'de', 'es', 'fr']


def check_dir(path):
    if os.path.isdir(path):
        print('Directory exists')
    else:
        print('The directory does not exist.')
        print('Create directory')
        os.mkdir(path)


def _path(lang, mode):
    return 'translated/' + mode + '_' + lang + '.csv'


def metric_f1(labels, preds):
    return f1_score(labels, preds, average='macro')


def run_model(model_name, model_type, params):
    print('model_name:', model_name)
    print('model_type:', model_type)
    lang_dict = {'en': 'English', 'de': 'German',
                 'es': 'Spanish', 'fr': 'French'}
    pprint.pprint(params)
    for i in range(4):
        lang = languages[i]
        params['output_dir'] = 'models/' + model_type + '-' + lang + '/'
        # modelを書き出すのでそこで推論と分けられます。
        print('train', lang_dict[lang])
        train_path = _path(lang, 'train')
        train_data = pd.read_csv(train_path)
        train_data['jobflag'] -= 1
        train, val = train_test_split(
            train_data, test_size=0.3, random_state=i)

        # ↑の切り方は完全にミスで、別コードを再利用してる時に間違えました。。。。。ひどい。。。。

        model = ClassificationModel(model_name, model_type, num_labels=4, weight=[
                                    1.17, 2.10, 0.532, 1.25], args=params, use_cuda=False)

        # weightはsklearn.utils.class_weight.compute_class_weightを使って計算しました

        model.train_model(train)
        losses, model_outputs, _ = model.eval_model(
            val, f1=metric_f1)
        print('example :')
        pprint.pprint(np.argmax(model_outputs, axis=1)[:10] + 1)
        # 1クラスに引っ張られがちだったので確認用です
        # ax = sns.countplot(np.argmax(model_outputs, axis=1) + 1)
        # fig = ax.get_figure()
        # fig.savefig(lang + '.png')
        # 分布確認用.
        print('loss:')
        pprint.pprint(losses)
        for lang2 in languages:
            print('predict', lang_dict[lang2])
            test_path = _path(lang2, 'test')
            test = pd.read_csv(test_path)[:2]
            y_pred, _ = model.predict(test['description'])
            result = pd.DataFrame({'jobflag': y_pred + 1})
            result.to_csv('result/' + model_name +
                          '_' + lang + '_' + lang2 + '.csv', index=False, header=False)


if __name__ == '__main__':
    check_dir('models')
    check_dir('result')
    params = yaml.load(open('config/params.yml'))
    run_model('bert', 'bert-base-cased', params)
    run_model('roberta', 'roberta-base', params)
    params['train_batch_size'] = 8
    params['eval_batch_size'] = 8
    run_model('xlnet', 'xlnet-base-cased', params)
    files = glob.glob('result/*.csv')
    results = pd.read_csv(files[0], header=None)
    id = range(2931, 4674)
    for file in files[1:]:
        result = pd.read_csv(file, header=None)
        # ax = sns.countplot(x=0, data=result)
        # fig = ax.get_figure()
        # fig.savefig('../visualize/' + file.split(".")[-2].split("/")[-1] + ".png")
        results = pd.concat([results, result], axis=1)

    submit = pd.DataFrame(
        {'id': id, 'jobflag': results.mode(axis=1)[0].astype(int)})
    submit.to_csv("submit/submit.csv", index=False, header=False)
