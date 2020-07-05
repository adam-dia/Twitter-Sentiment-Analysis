# Twitter Sentiment Analysis - Big Data Project - MIDITAL
# Team 3
# Team members - Adam Diarassouba, Gabrielle Otto, Michelle Tek, Snehajeet Chatterjee, Stella Schick

import re
import string 
import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from random import shuffle
from nltk import classify
from nltk import NaiveBayesClassifier
from collections import defaultdict

# First thing to do Installing NLTK
nltk.download()

# Secondly download Twitter_sample
nltk.download('twitter_samples')

# Finally we download the stopwords
nltk.download('stopwords')

# The twitter_samples corpus contains 3 files: 5000 positive tweets, 5000 negative tweets and 20.000 positive and negative tweets
# For this project, we will only be using a 10.000  dataset "twitter_sample" already 
# available in the "nltk.corpus module" i.e., the files of the 5000 postive and 5000 negative tweets

print ("Different type of tweet =>",twitter_samples.fileids())
pos_tweets = twitter_samples.strings('positive_tweets.json')
print ("Len of POSITIVE tweet",len(pos_tweets)) #output : 5000
neg_tweets = twitter_samples.strings('negative_tweets.json') 
print ("Len of NEGATIVE tweet", len(neg_tweets)) #output : 5000
all_tweets = twitter_samples.strings('tweets.20150430-223406.json')
print ("Length of TOTAL tweet from tweets.20150430-223406.json", len(all_tweets)) #output : 20000
    
# We import the TweetTokenizer Module first and then tokenize(split text into list)
tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

# Denoise the tweet by removing $GE, $RT, hyperlink, #, words like a, and, the, is, are, etc, emoticones, 
# punctuations and then convert word to Stem/Base by using Porter Stemming algorithmz

stopwords_english = stopwords.words('english')
stemmer = PorterStemmer()

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])

#combine all emoticones together
emoticons = emoticons_happy.union(emoticons_sad)

#Function to remove noise from the tweet
def denoise_tweet(tweet):
    # remove stock market tickers such as $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    
    tweets_clean = []    
    for word in tweet_tokens:
        if (word not in stopwords_english and 
              word not in emoticons and 
                word not in string.punctuation):
            stem_word = stemmer.stem(word) 
            tweets_clean.append(stem_word)
            
    return tweets_clean
custom_tweet = "RT @Twitter @CoronaVirus Hello There! Stay indoors. :) #safe #survive http://who.com"
# print cleaned tweet
print("Denoise the custom tweet ",denoise_tweet(custom_tweet)) #output: ['hello', 'stay', 'indoor', 'safe', 'surviv']

# feature extractor function
def bag_of_words(tweet):
    words = denoise_tweet(tweet)
    words_dictionary = dict([word, True] for word in words)    
    return words_dictionary

custom_tweet = "RT @Twitter @CoronaVirus Hello There! Stay indoors. :) #safe #survive http://who.com"
print (bag_of_words(custom_tweet)) #output: {'hello': True, 'stay': True, 'indoor': True, 'safe': True, 'surviv': True}

# positive tweets feature set
pos_tweets_set = []
for tweet in pos_tweets:
    pos_tweets_set.append((bag_of_words(tweet), 'pos'))    
 
# Now let's create a Train and Test Set
# We have 5000 positive tweets set and 5000 negative tweets set. We take 20% (i.e. 1000) 
# of positive tweets and 20% (i.e. 1000) of negative tweets as the test set. The remaining 
# negative and positive tweets will be taken as the training set.
# radomize pos_reviews_set and neg_reviews_set
# doing so will output different accuracy result everytime we run the program

neg_tweets_set = []
for tweet in neg_tweets:
    neg_tweets_set.append((bag_of_words(tweet), 'neg'))
    
shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
print(len(test_set), len(train_set)) # Output: (2000, 8000)

# Training Classifier and Calculating the Accuracy
# We train Naive Bayes Classifier using the training set and calculate the classification 
# accuracy of the trained classifier using the test set.
# First of all, we import the necessary modules classify and NaiveBayesClassifier in the nltk module
classifier = NaiveBayesClassifier.train(train_set)
accuracy = classify.accuracy(classifier, test_set)
#print(accuracy) # Output: 0.765
print ("************",classifier.show_most_informative_features(10)) 

# Now we test classifier with a custom tweet
# we provide a custom tweet and check the classification output of the trained classifier. 
# The classifier correctly predict both the negative and positive tweets provided 
custom_tweet = "We hate conora pandemic. It caused a lot of disaster."
custom_tweet_set = bag_of_words(custom_tweet)
print (classifier.classify(custom_tweet_set)) #output : neg for negative sentiment

# probability result
prob_result = classifier.prob_classify(custom_tweet_set)
print (prob_result) 
print (prob_result.max()) 
print (prob_result.prob("neg")) 
print (prob_result.prob("pos")) 

custom_tweet = "The medical corp is playing an amazing role. They're doing lot sacrifice"
custom_tweet_set = bag_of_words(custom_tweet)
 
print (classifier.classify(custom_tweet_set))

# probability result
prob_result = classifier.prob_classify(custom_tweet_set)
print (prob_result) 
print (prob_result.max()) 
print (prob_result.prob("neg")) 
print (prob_result.prob("pos"))

# Precision, Recall & F1-Score
# We load defaultdic submodule from collections module
# Accuracy is (correctly predicted observation) / (total observation)
#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#Precision = (TP) / (TP + FP)
#Recall = (TP) / (TP + FN)
#F1 Score = 2 * (precision * recall) / (precision + recall)

actual_set = defaultdict(set)
predicted_set = defaultdict(set)
 
actual_set_cm = []
predicted_set_cm = []
 
for index, (feature, actual_label) in enumerate(test_set):
    actual_set[actual_label].add(index)
    actual_set_cm.append(actual_label)
 
    predicted_label = classifier.classify(feature)
 
    predicted_set[predicted_label].add(index)
    predicted_set_cm.append(predicted_label)
    
from nltk.metrics import precision, recall, f_measure, ConfusionMatrix

#print('pos precision:', precision(actual_set['pos'], predicted_set['pos'])) 
#print('pos recall:', recall(actual_set['pos'], predicted_set['pos']))
#print('pos F-measure:', f_measure(actual_set['pos'], predicted_set['pos'])) 
#print('neg precision:', precision(actual_set['neg'], predicted_set['neg']))
#print('neg recall:', recall(actual_set['neg'], predicted_set['neg'])) 
#print('neg F-measure:', f_measure(actual_set['neg'], predicted_set['neg']))

# Confusion Matrix is a table we will be using to to describe the performance of the classifier
print('*****************| Performance of the classifier |*******************')
cm = ConfusionMatrix(actual_set_cm, predicted_set_cm)
print (cm)

# Result in tables print (cm) will look like
# – 761 negative tweets were correctly classified as negative (TN)
# – 239 negative tweets were incorrectly classified as positive (FP)
# – 231 positive tweets were incorrectly classified as negative (FN)
# – 769 positive tweets were correctly classified as positive (TP)

# Let's print the percentage of the performance
print('*****************| Percentage of the performance |*******************')
print (cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
# – 38.0% negative tweets were correctly classified as negative (TN)
# – 11.9% negative tweets were incorrectly classified as positive (FP)
# – 11.6% positive tweets were incorrectly classified as negative (FN)
# – 38.5% positive tweets were correctly classified as positive (TP)
