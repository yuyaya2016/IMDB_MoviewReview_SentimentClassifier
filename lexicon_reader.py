import os

POS_LABEL = '1'
NEG_LABEL = '0'


def check_if_exist(file_path):
    if not os.path.exists(file_path):
        print(file_path + ' could not be found')
        return False
    return True


def extract_word(word):
    return word.lower() if word.find('#') < 0 else word[:word.find('#')].lower()


def read_inqtabs(input_file_path):
    """
    :param input_file_path:
    :return lexicons: dictionary of labels (e.g. lexicons['good']: 1, lexicons['bad']: 0)
    """
    if not check_if_exist(input_file_path):
        return

    lexicons = dict()
    with open(input_file_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            elements = line.strip().split('\t')
            word = extract_word(elements[0])
            if len(word) > 0 and (elements[2] == 'Positiv' or elements[3] == 'Negativ'):
                label = POS_LABEL if elements[2] == 'Positiv' else NEG_LABEL
                lexicons[word] = label
    return lexicons


def read_senti_word_net(input_file_path):
    """
    :param input_file_path:
    :return lexicon: dictionary of lists (e.g. lexicons['good'][0]: positive score, lexicons['bad'][1]: negative score)
    """
    if not check_if_exist(input_file_path):
        return

    all_lexicons = dict()
    with open(input_file_path, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            if line.startswith('#'):
                continue

            elements = line.strip().split('\t')
            if len(elements) < 5 or len(elements[4]) == 0:
                continue

            for tmp_word in elements[4].split(' '):
                word = extract_word(tmp_word).replace('_', ' ')
                if len(word) > 0 and len(elements[2]) > 0 and len(elements[3]) > 0:
                    if word not in all_lexicons.keys():
                        all_lexicons[word] = list()
                        all_lexicons[word].append(list())
                        all_lexicons[word].append(list())
                    all_lexicons[word][0].append(float(elements[2]))
                    all_lexicons[word][1].append(float(elements[3]))

    lexicons = dict()
    for word in all_lexicons.keys():
        lexicons[word] = (max(all_lexicons[word][0]), max(all_lexicons[word][1]))
    return lexicons
