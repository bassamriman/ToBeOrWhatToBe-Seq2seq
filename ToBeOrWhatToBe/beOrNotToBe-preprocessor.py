from __future__ import print_function

import argparse
import os
import re

import nltk.data

parser = argparse.ArgumentParser()
parser.add_argument('--source-dir', help="Source data directory", default="./dataset/original_data")
parser.add_argument('--target-dir', help="Traget data directory", default="./dataset/pre_processed_data")
args = parser.parse_args()

be_verb_form = "am are were was is been being be".split()


def load_text():
    path = args.source_dir
    directory = os.fsencode(path)
    accumulated_text = ""

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        print(filename)
        if filename.startswith("wiki_"):
            file = open(path + "/" + filename, "r", encoding='utf-8')
            accumulated_text = accumulated_text + (file.read().replace('\n', ''))
            continue
        else:
            continue

    text = re.sub('<[^<]+>', "", accumulated_text)
    return text


def split_sentences(text): return nltk.sent_tokenize(text)


def split_words(text):
    tokens = nltk.word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return words


def split_words_regexp(text):
    pattern = r'''\n'''
    return nltk.regexp_tokenize(text, pattern)


def to_example(words):
    # words that are not be verbs
    new_context = []

    # be verbs in sentence
    be_verbs = []

    # split "be" verbs and other words
    for x in words:
        if x in be_verb_form:
            new_context.append("----")
            be_verbs.append(x)
        else:
            new_context.append(x)

    if not be_verbs:
        return ""

    # make one example per be verb in sentence
    return " ".join(new_context), " ".join(be_verbs)


def generate_dataset(text, root):
    sentences = split_sentences(text)

    # generate data file
    data_path = os.path.join(root, 'data.txt')
    with open(data_path, mode='w', encoding='utf-8') as fout:
        for sentence in sentences:
            words = split_words(sentence)
            new_example = to_example(words)
            if new_example == "":
                continue
            else:
                list1 = list(new_example)
                fout.write("\t".join(list1))
                fout.write('\n')


if __name__ == '__main__':
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    generate_dataset(load_text(), target_dir)
