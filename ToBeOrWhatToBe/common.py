import nltk.data


def sentence_to_words(sentence):
    split_sentence = sentence.split()

    # this is an intermediate step to prevent the word tokenizer form replacing "----" by two "--"
    tokanized_sentence_hyphens_to_star = ""
    for a in split_sentence:
        if a == "----":
            tokanized_sentence_hyphens_to_star = tokanized_sentence_hyphens_to_star + " " + "*"
        else:
            tokanized_sentence_hyphens_to_star = tokanized_sentence_hyphens_to_star + " " + a

    seq = nltk.word_tokenize(tokanized_sentence_hyphens_to_star)
    words = []
    for a in seq:
        if a == "*":
            words.append("----")
        else:
            words.append(a)

    return words
