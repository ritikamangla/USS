import speech_recognition as sr
from speech_recognition import Recognizer

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

# If you get an error uncomment this line and download the necessary libraries
# nltk.download()
"""r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything :")
    audio = r.listen(source)
    try:
        print('recognizing....')
        text = r.recognize_google(audio)
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize what you said")
"""
text=input("ENTER ARTICLE:")
stemmer = SnowballStemmer("english")
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue

    word = stemmer.stem(word)

    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = sent_tokenize(text)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

# Average value of a sentence from original text
average = int(sumValues / len(sentenceValue))
# print("average:",average)

summary = ''
if len(text) > 700:
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            #  print(sentenceValue[sentence])
            summary += " " + sentence
else:
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (0.8* average)):
            #  print(sentenceValue[sentence])
            summary += " " + sentence
#print(len(text))
print(summary)
#print(len(summary))