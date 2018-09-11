# To be or what to be, that is the question
This is a solution to the [problem](https://www.hackerrank.com/challenges/to-be-what/problem) listed on HackerRank.

This solution uses encoder decoder LSTM neural networks implemented using IBM's pytorch implemation for [seq2seq](https://github.com/IBM/pytorch-seq2seq).

Main libraries used:
* [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq) for encoder decoder LSTM implementation
* [torchtext](https://github.com/pytorch/text) for data loading, batching and vocabulary labeling
* [pytorch](https://github.com/pytorch/pytorch) for tensor computation with GPU acceleration and deep neural network.
* [nltk](https://www.nltk.org) for training data pre-processing.