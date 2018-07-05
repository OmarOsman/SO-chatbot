
# coding: utf-8

# # Final project: StackOverflow assistant bot
# 
# Congratulations on coming this far and solving the programming assignments! In this final project, we will combine everything we have learned about Natural Language Processing to construct a *dialogue chat bot*, which will be able to:
# * answer programming-related questions (using StackOverflow dataset);
# * chit-chat and simulate dialogue on all non programming-related questions.
# 
# For a chit-chat mode we will use a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot).
# Those who aim at honor certificates for our course or are just curious, will train their own models for chit-chat.
# ![](https://imgs.xkcd.com/comics/twitter_bot.png)
# ©[xkcd](https://xkcd.com)

# ### Data description
# 
# To detect *intent* of users questions we will need two text collections:
# - `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (*positive samples*).
# - `dialogues.tsv` — dialogue phrases from movie subtitles (*negative samples*).
# 

# 
# For those questions, that have programming-related intent, we will proceed as follows predict programming language (only one tag per question allowed here) and rank candidates withing the tag using embeddings.
# For the ranking part, you will need:
# - `word_embeddings.tsv` — word embeddings, that you  trained with StarSpace in the 3rd assignment. It's not a problem if you didn't do it, because we can offer an alternative solution for you.

# As a result of this notebook, you should obtain the following new objects that you will then use in the running bot:
# 
# - `intent_recognizer.pkl` — intent recognition model;
# - `tag_classifier.pkl` — programming language classification model;
# - `tfidf_vectorizer.pkl` — vectorizer used during training;
# - `thread_embeddings_by_tags` — folder with thread embeddings, arranged by tags.
#     

# Some functions will be reused by this notebook and the scripts, so we put them into *utils.py* file. Don't forget to open it and fill in the gaps!

# In[1]:


from utils import *


# ## Part I. Intent and language recognition

# We want to write a bot, which will **answer programming-related questions**, but also will be able to **maintain a dilogue**. We would also like to detect the *intent* of the user from the question (we could have had a 'Question answering mode' check-box in the bot, but it wouldn't fun at all, would it?). So the first thing we need to do is to **distinguish programming-related questions from general ones**.
# 
# It would also be good to predict which programming language a particular question referres to. By doing so, we will speed up question search by a factor of the number of languages (10 here), and exercise our *text classification* skill a bit. :)

# In[2]:


import numpy as np
import pandas as pd
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer


# ### Data preparation

# In the first assignment (Predict tags on StackOverflow with linear models), you have already learnt how to preprocess texts and do TF-IDF tranformations. Reuse your code here. In assition, you will also need to [dump](https://docs.python.org/3/library/pickle.html#pickle.dump) the TF-IDF vectorizer with pickle to use it later in the running bot.

# In[3]:


def tfidf_features(X_train, X_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    
    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.
    
    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.
    
    tfidf_vectorizer = TfidfVectorizer(input = X_train , token_pattern='(\S+)',
                                       ngram_range=(1, 2), max_df=0.9, min_df=5)
    tfidf_vectorizer.fit(X_train)
    

    pickle_out = open(vectorizer_path,"wb")
    pickle.dump(tfidf_vectorizer, pickle_out)
    pickle_out.close()
    
    X_train = tfidf_vectorizer.transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    
    
    return X_train, X_test


# Now, load examples of two classes. Use a subsample of stackoverflow data to balance the classes. You will need the full data later.

# In[4]:


sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)


# Check how the data look like:

# In[5]:


#dialogue_df.head()


# In[6]:


#stackoverflow_df.head()


# Apply *text_prepare* function to preprocess the data:

# In[7]:


dialogue_df['text'] = dialogue_df.text.apply(lambda x : text_prepare(x))
stackoverflow_df['title'] = stackoverflow_df.title.apply(lambda x : text_prepare(x))


# ### Intent recognition

# We will do a binary classification on TF-IDF representations of texts. Labels will be either `dialogue` for general questions or `stackoverflow` for programming-related questions. First, prepare the data for this task:
# - concatenate `dialogue` and `stackoverflow` examples into one sample
# - split it into train and test in proportion 9:1, use *random_state=0* for reproducibility
# - transform it to TF-IDF features

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state = 0)
#print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_list = [x for x in X_train]
X_test_list = [x for x in X_test]
X_train_tfidf, X_test_tfidf = tfidf_features(X_train_list,X_test_list,RESOURCE_PATH['TFIDF_VECTORIZER'])


# Train the **intent recognizer** using LogisticRegression on the train set with the following parameters: *penalty='l2'*, *C=10*, *random_state=0*. Print out the accuracy on the test set to check whether everything looks good.

# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[11]:


lg = LogisticRegression(penalty='l2',C=10,random_state=0)
intent_recognizer = lg.fit(X_train_tfidf,y_train)


# In[12]:


# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
#print('Test accuracy = {}'.format(test_accuracy * 100))


# Dump the classifier to use it in the running bot.

# In[13]:


pickle.dump(intent_recognizer, open(RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))


# ### Programming language clasification 

# We will train one more classifier for the programming-related questions. It will predict exactly one tag (=programming language) and will be also based on Logistic Regression with TF-IDF features. 
# 
# First, let us prepare the data for this task.

# In[14]:


X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))


# Let us reuse the TF-IDF vectorizer that we have already created above. It should not make a huge difference which data was used to train it.

# In[16]:


vectorizer = pickle.load(open(RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)


# Train the **tag classifier** using OneVsRestClassifier wrapper over LogisticRegression. Use the following parameters: *penalty='l2'*, *C=5*, *random_state=0*.

# In[17]:


from sklearn.multiclass import OneVsRestClassifier


# In[18]:


ov = OneVsRestClassifier(LogisticRegression(penalty='l2',C=5,random_state=0))
tag_classifier = ov.fit(X_train_tfidf,y_train)


# In[19]:


# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy*100))


# Dump the classifier to use it in the running bot.

# In[20]:


pickle.dump(tag_classifier, open(RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))


# ## Part II. Ranking  questions with embeddings

# To find a relevant answer (a thread from StackOverflow) on a question you will use vector representations to calculate similarity between the question and existing threads. We already had `question_to_vec` function from the assignment 3, which can create such a representation based on word vectors. 
# 
# However, it would be costly to compute such a representation for all possible answers in *online mode* of the bot (e.g. when bot is running and answering questions from many users). This is the reason why you will create a *database* with pre-computed representations. These representations will be arranged by non-overlaping tags (programming languages), so that the search of the answer can be performed only within one tag each time. This will make our bot even more efficient and allow not to store all the database in RAM. 

# Load StarSpace embeddings which were trained on Stack Overflow posts. These embeddings were trained in *supervised mode* for duplicates detection on the same corpus that is used in search. We can account on that these representations will allow us to find closely related answers for a question. 
# 
# If for some reasons you didn't train StarSpace embeddings in the assignment 3, you can use [pre-trained word vectors](https://code.google.com/archive/p/word2vec/) from Google. All instructions about how work with these vectors were provided in the same assignment.

# In[21]:


#wv_embeddings = KeyedVectors.load_word2vec_format( r'C:\Users\Omar\AnacondaProjects\natural-language-processing-master\week3\Word2Vec\GoogleNews-vectors-negative300.bin',
                                           #binary=True,limit=500000,unicode_errors='ignore')
starspace_embeddings, embeddings_dim = load_embeddings('data/starspace_embedding.tsv')


# Since we want to precompute representations for all possible answers, we need to load the whole posts dataset, unlike we did for the intent classifier:

# In[44]:


posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')


# In[45]:


#posts_df.head()


# Look at the distribution of posts for programming languages (tags) and find the most common ones. 
# You might want to use pandas [groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) and [count](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.count.html) methods:

# In[46]:


posts_df['tag'] = posts_df['tag'].replace('c\c++', 'c-c++')


# In[47]:


counts_by_tag = posts_df.groupby('tag').count()['post_id']
#counts_by_tag


# In[48]:


for tag, count in counts_by_tag.items():
        print (tag,count)


# In[51]:


counts_by_tag['ruby']


# In[52]:


tag_posts = posts_df[posts_df['tag'] == 'ruby']
#len(list(tag_posts['post_id']))
 


# Now for each `tag` you need to create two data structures, which will serve as online search index:
# * `tag_post_ids` — a list of post_ids with shape `(counts_by_tag[tag],)`. It will be needed to show the title and link to the thread;
# * `tag_vectors` — a matrix with shape `(counts_by_tag[tag], embeddings_dim)` where embeddings for each answer are stored.
# 
# Implement the code which will calculate the mentioned structures and dump it to files. It should take several minutes to compute it.

# In[53]:


import os
os.makedirs(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)

for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]
    
    tag_post_ids = list(tag_posts['post_id'])
    
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = question_to_vec(title,wv_embeddings,embeddings_dim)

    # Dump post ids and vectors to a file.
    filename = os.path.join(RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], '%s.pkl' % tag)
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))

