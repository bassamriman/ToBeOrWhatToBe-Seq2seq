import argparse
import logging

import nltk.data

from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from common import sentence_to_words

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--text-path', action='store', dest='text_path', default="./text-to-predict.txt",
                    help='Path to the text file that contrain sentences with missing verb "Be"')
parser.add_argument('--trained-model-dir', action='store', dest='trained_model_dir', default='./trained_model',
                    help='Path to trained model directory')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

args = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, args.log_level.upper()))
logging.info(args)

logging.info("loading checkpoint from {}".format(args.trained_model_dir))
checkpoint_path = args.trained_model_dir
checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

predictor = Predictor(seq2seq, input_vocab, output_vocab)

with open(args.text_path, mode='r', encoding='utf-8') as file:
    file.readline()
    text = file.read().replace('\n', '')
    sentences = nltk.sent_tokenize(text)
    results = []
    for sentence in sentences:
        words = sentence_to_words(sentence)

        result = predictor.predict(words)
        result.remove('<eos>')
        if result:
            results.extend(result)
    print("\n".join(results))
