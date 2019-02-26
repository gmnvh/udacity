# Lesson 11 - Text Learning

- Fundamental question about text learning: What is the input feature ?
	- The length of e-mail or text changes so can not be used as input
	- bag of words: use frequency of words known based a dictionary
		- order does not matter
		- long phrases give different input vectors
		- can not handle complex phrases like "chicago bulls"
		- sklearn: CountVectorizer (L11Q6)
		- stopwords: low information words that happen very frequently
		- stemmer:  group together same meaning words

- Order of Operations in Text Processing
	- stemming
	- bag of words

- TfIdf Representation
	- Tf - term frequency - like bag of words
	- Idf - inverse document frequency - weighting by how often word occurs in corpus