# 翻訳するやつです。
# 何も考えずにgoogle transを使いましたが、他にも選択肢があった気がします.....(すごい遅いです)

import pandas as pd
import os
from googletrans import Translator
from tqdm import tqdm

tqdm.pandas()


def check_dir(path):
    if os.path.isdir(path):
        print('Directory exists')
    else:
        print('The directory does not exist.')
        print('Create directory')
        os.mkdir(path)



def safe_eng_translate(text):
    """
    たまにうまく翻訳できないことがあるので、信頼性を担保するためにこれを通しています
    """
    lang = translator.detect(text).lang
    if lang != 'en':
        return safe_eng_translate(translator.translate(text, src=lang).text)
    else:
        return text


def trans(text, lang):
    result = translator.translate(translator.translate(
        text, dest=lang).text, src=lang).text
    return safe_eng_translate(result)


def processing(mode, lang):
    print('Start translation', mode)
    print('lang :', lang)
    def _trans(text): return trans(text, lang)
    file_name = data_storage_path + mode + '_' + lang + '.csv'
    transed = dict[mode]['description'].progress_apply(_trans)
    transed = pd.DataFrame(transed)
    print('Finish translation.')
    print('Example')
    print('----------------------')
    print(transed[:10])
    print('----------------------')
    if mode == 'train':
        result = pd.DataFrame(
            {'description': transed['description'], 'jobflag': train['jobflag']})
        result.to_csv(file_name, index=False)
    elif mode == 'test':
        result = pd.DataFrame(
            {'description': transed['description']})
        result.to_csv(file_name, index=False)
    print('Export data as', file_name)
    print('----------------------')




data_storage_path = 'translated/'
print('Check for the existence of the directory to be stored....')
check_dir(data_storage_path)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('Export the original file')
train.drop('id', axis=1).to_csv('translated/train_en.csv', index=False)
test.drop('id', axis=1).to_csv('translated/test_en.csv', index=False)


dict = {
    'train': train,
    'test': test
}


translator = Translator()
languages = ['de', 'es', 'fr']
# ドイツ、スペイン、フランス




for lang in languages:
    processing('train', lang)
    processing('test', lang)


print('run successfully')
