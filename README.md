# USS
Introduction:
We propose an unsupervised text summarization approach by clustering sentence embeddings trained to embed paraphrases near each other. Clusters of sentences are then converted to a summary by selecting a representative from each cluster.When representing sentences in a high-dimensional vector space, the goal is typically to directly or indirectly embed sentences such that sentences close in meaning are embedded near each other in the vector space. Thus, sentences that form a cluster in the vector space are likely to be close in meaning to each other. We exploit this assumption to perform summarization.


Step 1: Sentence Tokenization
Natural Language toolkit has very important module tokenize which further compromises of sub-modules:
word tokenize
sentence tokenize
After converting speech to text, we will split the text into sentences based on specific rules of sentence delimiters.

Step 2: Create Sentence Embeddings
This work  aims to learn how to combine word embeddings to obtain sentence embeddings that satisfy the property that sentences that are paraphrases of each other are embedded near each other in the vector space. This is done in a supervised manner using known paraphrases. The authors compare different technqiues for combining word embeddings and test the learned embeddings on prediction of textual similarity and entailment, and in sentiment classification. They find that averaging word embeddings learned in a supervised manner performs best for prediction of textual similarity and entailment. 
For sentence embeddings, one easy way is to take a weighted sum of the word vectors for the words contained in the sentence. We take a weighted sum because frequently occurring words such as ‘and’, ‘to’ and ‘the’, provide little or no information about the sentence. Some rarely occurring words, which are unique to a few sentences have much more representative power. Hence, we take the weights as being inversely related to the frequency of word occurrence.  
However, these unsupervised methods do not take the sequence of words in the sentence into account. This may incur undesirable losses in model performance. To overcome this, I chose to instead train a Skip-Thought sentence encoder in a supervised manner using Wikipedia dumps as training data. The Skip-Thoughts model consists of two parts:
Encoder Network: The encoder is typically a GRU-RNN which generates a fixed length vector representation h(i) for each sentence S(i)in the input. The encoded representation h(i) is obtained by passing final hidden state of the GRU cell (i.e. after it has seen the entire sentence) to multiple dense layers.
Decoder Network: The decoder network takes this vector representation h(i) as input and tries to generate two sentences — S(i-1)and S(i+1), which could occur before and after the input sentence respectively. Separate decoders are implemented for generation of previous and next sentences, both being GRU-RNNs. The vector representation h(i) acts as the initial hidden state for the GRUs of the decoder network.

Step 3: Clustering
K-means clustering is a clustering algorithm that aims to partition n observations into k clusters.
There are 3 steps:
Initialisation – K initial “means” (centroids) are generated at random
Assignment – K clusters are created by associating each observation with the nearest centroid
Update – The centroid of the clusters becomes the new mean
Assignment and Update are repeated iteratively until convergence
The end result is that the sum of squared errors is minimised between points and their respective centroids.
In this code , we chose  the total number of sentences in the summary to be equal to the square root of the maximum number of sentences in the document.The fit function is used to compute the k-means clustering.

Step 4: Summarization
Clustering is an unsupervised machine learning approach, but it be used to improve the accuracy of supervised machine learning algorithms as well by clustering the data points into similar groups and using these cluster labels as independent variables in the supervised machine learning algorithm.
Each cluster of sentence embeddings can be interpreted as a set of semantically similar sentences whose meaning can be expressed by just one candidate. The candidate sentence is chosen to be the sentence whose vector representation is closest to the cluster center. Candidate sentences corresponding to each cluster are then ordered to form a summary for the document.The order of the sentences in the summary is determined by the position of the sentences in the cluster.
