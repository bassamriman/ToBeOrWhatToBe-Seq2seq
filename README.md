# To be or what to be, that is the question
This is a solution to the [problem](https://www.hackerrank.com/challenges/to-be-what/problem) listed on HackerRank.

This solution uses encoder decoder LSTM neural networks implemented using IBM's pytorch implemation for [seq2seq](https://github.com/IBM/pytorch-seq2seq).

Main libraries used:
* [pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq) for encoder decoder LSTM implementation
* [torchtext](https://github.com/pytorch/text) for data loading, batching and vocabulary labeling
* [pytorch](https://github.com/pytorch/pytorch) for tensor computation with GPU acceleration and deep neural network.
* [nltk](https://www.nltk.org) for training data pre-processing.

##  Using beOrNotToBe-predictor.py to predict the correct form of verb "be" in a given text

1. Add your text inside text-to-predict.txt or specify the path to your own text when running beOrNotToBe-predictor.py.
2. Run beOrNotToBe-predictor.py 
    * Use --text-path your/path/here (to override the default path: ./text-to-predict.txt)
    
The input text has contain two lines. The first line will contain only one integer N, which will equal the number of blanks in the text. The second line contains one paragraph of text. Several occurrences of the words mentioned previously have been blanked out and replaced by four consecutive hyphens (----). These are the blanks which you need to fill up with one of the following words: 'am','are','were','was','is','been','being','be'
    
##  Using beOrNotToBe-preprocessor.py to pre-process your data for training

1. Add your text files to the directory path ./ToBeOrWhatToBe/dataset/original_data (one or many files containing text)
2. Run beOrNotToBe-preprocessor.py
    * Output will be generated in the following directory path ./ToBeOrWhatToBe/dataset/pre_processed_data.
    
##  Using beOrNotToBe-trainer.py to train the model
1. Make sure you your pre-process your training data using beOrNotToBe-preprocessor.py first.
2. Run beOrNotToBe-trainer.py
    * Add argument --resume to resume from last the checkpoint should the training get interrupted.
    * Add argument --load_checkpoint 2018_09_08_18_34_58 to reach a prompt that will allow you to input text and get a prediction
       from the trained model loaded from the specified checkpoint.

