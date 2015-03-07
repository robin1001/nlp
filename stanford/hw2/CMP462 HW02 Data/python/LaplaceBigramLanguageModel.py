import math, collections


class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    # TODO your code here
    self.bigram_dic = collections.defaultdict(lambda: 0)
    self.unigram_dic = collections.defaultdict(lambda: 0)
    self.train(corpus)

    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    for sentence in corpus.corpus:
      for i in range(0, len(sentence.data)-1):
        cur = sentence.data[i].word
        next = sentence.data[i+1].word
        self.unigram_dic[cur] = self.unigram_dic[cur]+1
        self.bigram_dic[(cur, next)] = self.bigram_dic[(cur, next)] + 1
    
  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score=0
    for i in range(1, len(sentence)-1):
      cur = sentence[i]
      pre = sentence[i-1]
      #print pre, cur 
      total = self.unigram_dic[pre] 
      count = self.bigram_dic[(pre, cur)] + 1
      score += math.log(count)
      score -= math.log(total + len(self.unigram_dic))
    return score
