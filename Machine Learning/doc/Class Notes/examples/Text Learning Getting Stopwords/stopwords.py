import nltk
from nltk.corpus import stopwords

try:
    sw = stopwords.words("english")
    
    print sw[0]
    print sw[10]
    
    print "Size of stopwords: ", len(sw)
except:
    nltk.download()
    