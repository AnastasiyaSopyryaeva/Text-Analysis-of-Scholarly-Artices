import pandas as pd
import matplotlib.pyplot as plt
import json
from collections import Counter
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud

# STEP 1 Reading the file
# Reading JSTOR JSONL file and transforming it into a list of JSON objects (documents of the corpus)
data = []
with open('health.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# STEP 2 Dataset cleaning
# Filtering data to leave documents with the most relevant articles: containing key words in title, abstract or keyphrase list
filtered_data = []
for object in data:
    if 'keyphrase' in object and object not in filtered_data and ('homosexual' in object['keyphrase'] or 'sexual orientation' in object['keyphrase'] or 'sexuality' in object['keyphrase'] or 'homosexuality' in object['keyphrase']):
        filtered_data.append(object)
    if 'title' in object and object not in filtered_data and ('homosexual' in object['title'] or 'sexual orientation' in object['title'] or 'sexuality' in object['title'] or 'homosexuality' in object['title']):
        filtered_data.append(object)
    if 'abstract' in object and object not in filtered_data and ('homosexual' in object['abstract'] or 'sexual orientation' in object['abstract'] or 'sexuality' in object['abstract'] or 'homosexuality' in object['abstract']):
        filtered_data.append(object)

# Removing from the dataset articles that duplicates in other datasets (in other disciplines). 
# The duplicates id's were discovered in advance
list_of_duplicates = ['ark://27927/phx5xxn90bt', 'ark://27927/phz1287t0vj', 'ark://27927/phx85mnh8cc', 
'ark://27927/phxp05gjq4', 'ark://27927/phz6qvtmhrb', 'ark://27927/pgkz9ck7nc', 'ark://27927/pgk3swjqrnm', 
'ark://27927/phx81qz8bfw', 'ark://27927/phz2js5vfzc', 'ark://27927/phz6302zv3t', 'ark://27927/phxg4zz2g0', 
'ark://27927/phx5z6ztx8n', 'ark://27927/pgk28cpw7dz', 'ark://27927/pgf8vtf8c7', 'ark://27927/phz2mjtjc8b', 
'ark://27927/pgg1pr2qkgk', 'ark://27927/phx5xxpfgw8', 'ark://27927/pgj296tq0r7', 'ark://27927/phx6x89dqmj', 
'ark://27927/phw8gj2rwn', 'ark://27927/phws3wp75', 'ark://27927/phx4nfkkpxb', 'ark://27927/phx7wk9rp7g', 
'ark://27927/pgk2k5gpq67', 'ark://27927/pgh31p1qb4z', 'ark://27927/phxcj3zzk8', 'ark://27927/phx564fpcc6', 
'ark://27927/phz31zbvccx', 'ark://27927/phzgbnjjk1b', 'ark://27927/phz809dwxvs', 'ark://27927/phz2md7qq4f', 
'ark://27927/phx2p42krdb', 'ark://27927/pgk2bm3d43q', 'http://www.jstor.org/stable/44969634']
filtered_data = [i for i in filtered_data if not (i['id'] in list_of_duplicates)]

# Converting list of JSON's to pandas dataframe 
df_health = pd.DataFrame(filtered_data)

# Removing irrelevant for the study columns
df_health = df_health[['id', 'isPartOf', 'keyphrase', 'pageCount', 'publicationYear', 'publisher', 'title', 'wordCount', 'unigramCount', 'bigramCount', 'trigramCount', 'abstract', 'creator']]

# STEP 3 Data Analysis

# 1) Unigrams analysis. Creating bag-of-words with unigrams and look at most common
unigrams_list = []
for i in range(len(filtered_data)):
    unigrams_list.append(filtered_data[i]['unigramCount'])

# cleaning unigrams list
stop_list_health = ['patients', 'heterosexual', 'rates', 'positive', 'negative', 'greater', 'survey', 'included', 'analysis', 'three', 'variables', 'respondents', 'population', 'including', 'number', 'substance', 'whether', 'important', 'model', 'sexually', 'current', 'findings', 'significant', 'general', 'toward', 'prevalence', 'based', 'compared', 'groups', 'years', 'report', 'attitudes', 'high', 'using', 'factors', 'related', 'table', 'individuals', 'clinical', 'associated', 'differences', 'higher', 'sample', 'scale', 'across', 'previous', 'total', 'potential', 'present', 'similar', 'provide', 'de', 'practice', 'increased', 'effects', 'given', 'levels', 'knowledge', 'identified', 'level', 'patient', 'role', 'likely', 'review', 'significantly', 'effect', 'evidence', 'past' 'group', 'participants', 'results', 'medical', 'et', 'c', 'b', 'e', 'n', 'p', 'r', 'j', 'homosexuality', 'reported', 'health', 'university', 'journal', 'issues', 'new', 'used', 'studies', 'percent', 'become', 'make', 'good', 'think', 'case', 'need', 'less', 'view', 'order', 'found', 'understanding', 'another', 'first', 'part', 'might', 'made', 'much', 'like', 'often', 'must', 'us', 'well', 'could', 'issue', 'human', 'see', 'may', 'even', 'people', 'would', 'also', 'although', 'many', 'orientation', 'author', 'essay', 'sexual', 'gay', 'gays', 'use', 'since', 'social', 'sexuality', 'article', 'within', 'lesbian', 'lesbians', 'study', 'research', 'among', 'homosexual', 'rather', 'thereby', 'paper', 'specific', 'fully', 'yet', 'al','one', 'two', 'data']
processed_unigrams_list = []
for i in unigrams_list:
    for key, value in i.items():
        key = key.lower()
        if key not in stopwords.words('english') and key not in stop_list_health and key.isalpha():       
            processed_unigrams_list.append({key: value})

# summing unigrams across the whole dataset
unigrams_counter = Counter()            
for d in processed_unigrams_list:
   unigrams_counter.update(d)
print(unigrams_counter.most_common(100))

# WordCloud unigrams
wordcloud = WordCloud(width=900,height=500).generate_from_frequencies(unigrams_counter)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# 2) Topic Modeling
# token cleaner function
stop_words = stopwords.words('english')
def process_token(token):
    token = token.lower()
    if token in stop_words:
        return
    if token in stop_list_health:
        return
    if len(token) < 4:
        return
    if not(token.isalpha()):
        return
    return token

# unigrams transformation
limit = 3000
n = 0
documents = []
for document in filtered_data:
    processed_document = []
    document_id = document["id"]

    unigrams = document.get("unigramCount", {})
    for gram, count in unigrams.items():
        clean_gram = process_token(gram)
        if clean_gram is None:
            continue
        processed_document += [clean_gram] * count # Add the unigram as many times as it was counted
    if len(processed_document) > 0:
        documents.append(processed_document)
    if n % 1000 == 0:
        print(f'Unigrams collected for {n} documents...')
    n += 1
    if (limit is not None) and (n >= limit):
       break
print(f'All unigrams collected for {n} documents.')

# preparing bag of words for further processing
import gensim
dictionary = gensim.corpora.Dictionary(documents)
doc_count = len(documents)
num_topics = 5 # Check the number of topics
passes = 5 # The number of passes used to train the model
# Remove terms that appear in less than 50 documents and terms that occur in more than 90% of documents.
dictionary.filter_extremes(no_below=50, no_above=0.90)
bow_corpus = [dictionary.doc2bow(doc) for doc in documents]

# LDA model
model = gensim.models.LdaModel(
    corpus=bow_corpus,
    id2word=dictionary,
    num_topics=num_topics,
    passes=passes
)

# coherence score for the model check
from gensim.models import CoherenceModel
coherence_model_lda = CoherenceModel(
    model=model,
    corpus=bow_corpus,
    dictionary=dictionary, 
    coherence='u_mass'
)

coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)   # num_topics = 2, score = -0.3939, -0.3881; 
                                              # num_topics = 3, score = -0.4543, -0.4197; 
                                              # num_topics = 4, score = -0.4650, -0.3879; 
                                              # num_topics = 5, score = -0.4889, -0.5009; 
                                              # num_topics = 6, score = -0.4686, -0.4666;
                                              # num_topics = 7, score = -0.4823, -0.4942;
                                              # num_topics = 8, score = -0.4920, -0.4830.

# printing important words for each topic
for topic_num in range(0, num_topics):
    word_ids = model.get_topic_terms(topic_num)
    words = []
    for wid, weight in word_ids:
        word = dictionary.id2token[wid]
        words.append(word)
    print("Topic {}".format(str(topic_num).ljust(5)), " ".join(words))

# LDA model visualization
import pyLDAvis 
import pyLDAvis.gensim_models
visualisation = pyLDAvis.gensim_models.prepare(model, bow_corpus, dictionary)
pyLDAvis.save_html(visualisation, 'LDA_Visualization_h8.2.html')

# 3) Analysing publishing dates distribution
df_health.groupby(['publicationYear'])['id'].agg('count').plot.bar(title='Documents by year (health articles)', figsize=(20, 5), fontsize=12); 
plt.show()

# 4) Analysing page length distribution
plt.boxplot(x=df_health['pageCount'])
plt.suptitle('Page length (health articles)')
plt.show()

# 5) Analysing topic evolution by publication year 

df_health_year_less88 = df_health[df_health['publicationYear'] <= 1988]
df_health_year_less00 = df_health.query('publicationYear > 1988 and publicationYear <= 2000')
df_health_year_less07 = df_health.query('publicationYear > 2000 and publicationYear <= 2007')
df_health_year_less15 = df_health.query('publicationYear > 2007 and publicationYear <= 2015')
df_health_year_less22 = df_health[df_health['publicationYear'] > 2015]

unigrams_list = []
for i, v in df_health_year_less22.iterrows(): # change dataframe
    unigrams_list.append(v['unigramCount'])

unigrams_counter = Counter()
processed_unigrams_list = []
for i in unigrams_list:
    for key, value in i.items():
        key = key.lower()
        if key not in stopwords.words('english') and key not in stop_list_health and key.isalpha():       
            processed_unigrams_list.append({key: value})
            
for d in processed_unigrams_list:
   unigrams_counter.update(d)
print(unigrams_counter.most_common(100))