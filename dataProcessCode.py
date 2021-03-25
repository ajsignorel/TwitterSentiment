
#Utility Libaries
import re
import pandas as pd
from pandas import DataFrame

#Natural Language ToolKit
import nltk
#nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


columnsData  = ["sentiment", "ids", "date", "flag", "user", "text"]
encodingData = "ISO-8859-1"
data = pd.read_csv('C:/Datasets/TwitterData/training1600000.csv', encoding=encodingData , names=columnsData)

#Remove columns so data[] only has sentiment and text
#Removed Columns: ids, date, flag, and user
data = data[['sentiment','text']]

# Replace sentiment value of 4 to 1
data['sentiment'] = data['sentiment'].replace(4,1)

#Graphing the Data
ax = data.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)

#Sorting the list
text, sentiment = list(data['text']), list(data['sentiment'])

#Dictionary of emojis with their meaning
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


#List of Stopwords
swl = stopwords.words("english")
#print(stopwordlist)

def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replaces all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replaces all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            if word not in swl:
                if len(word)>1:
                #Lemmatizing the word.
                    word = wLemm.lemmatize(word)
                    tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText

import time
t = time.time()
processedtext = preprocess(text)
print(f'Processing Text Complete.')
print(f'Time: {round(time.time()-t)} seconds')
df1 = DataFrame (sentiment, columns=['sentiment'])
df2 = DataFrame (processedtext, columns=['text'])
processedData = df1.join(df2)
#print(processedData)
#processedData.to_csv(r'C:\Users\Owner\Dropbox\TwitterSentiment\processedData.csv', index=False, encoding='ISO-8859-1')

