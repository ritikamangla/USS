import numpy as np
from talon.signature.bruteforce import extract_signature
from langdetect import detect
from nltk.tokenize import sent_tokenize
import skipthoughts
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import speech_recognition as sr
from speech_recognition import Recognizer

"""#speech to text
r=sr.Recognizer() 
with sr.Microphone() as source:
    print("Say something...")
    audio = r.listen(source)
   
try:
    print("Recognizing..")
    text=r.recognize_google(audio,language='en-in')
    print("You said",text)

except Exception as e:
    print(e)
"""
#text=input("ENTER ARTICLE")
from google.cloud import speech_v1p1beta1 as speech
client = speech.SpeechClient()

speech_file = '/home/dhwani/Downloads/13(1).wav'

with open(speech_file, 'rb') as audio_file:
    content = audio_file.read()

audio = speech.types.RecognitionAudio(content=content)

config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    #sample_rate_hertz=8000,
    language_code='en-US',
    model='phone_call',
    enable_speaker_diarization=True,
    enable_word_time_offsets=True,
    diarization_speaker_count=2,
    enable_automatic_punctuation=True
    )

print('Waiting for operation to complete...')
response = client.recognize(config, audio)

    # [END speech_transcribe_auto_punctuation_beta]


# The transcript within each result is separate and sequential per result.
# However, the words list within an alternative includes all the words
# from all the results thus far. Thus, to get all the words with speaker
# tags, you only have to take the words list from the last result:
result = response.results[-1]

words_info = result.alternatives[0].words
speaker1_transcript=""
speaker2_transcript=""

# Printing out the output:
for word_info in words_info:
    if(word_info.speaker_tag==1): speaker1_transcript=speaker1_transcript+ word_info.word + ' '
    if(word_info.speaker_tag==2): speaker2_transcript=speaker2_transcript+ word_info.word + ' '
    
    #print("word: '{}', speaker_tag: {}".format(word_info.word,
    #                                           word_info.speaker_tag))
print("speaker1.'{}'".format(speaker1_transcript))
print("speaker2.'{}'".format(speaker2_transcript))
text=speaker1_transcript+speaker2_transcript
print("TEXT IS")
print(text)
print("Splitting into sentences....")

sentences=sent_tokenize(text)

for j in reversed(range(len(sentences))):
    sent = sentences[j]
    sentences[j] = sent.strip()
    
    if sent == '':
        sentences.pop(j)
text = sentences
 
enc_text=[None]

#all_sentences=sent_tokenize(text) #[sent for sent in text]
 

print('Loading pre-trained models...')
model = skipthoughts.load_model()
 
encoder = skipthoughts.Encoder(model)
print('Encoding sentences...')
 
enc_sentences = encoder.encode(text,verbose=False)
enc_text=enc_sentences

summary = [None]

n_clusters = int(np.ceil(len(enc_text)**0.8))

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans = kmeans.fit(enc_text)

avg = []
closest = []

for j in range(n_clusters):
    idx = np.where(kmeans.labels_ == j)[0]
    avg.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_,\
   	                                     enc_text)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])

summary = ' '.join(text[closest[idx]] for idx in ordering)

print('Clustering Finished')

print(summary)     

f= open("SUMMARY","w+")
f.write(summary)
f.close()