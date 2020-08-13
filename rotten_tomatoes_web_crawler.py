#!/usr/bin/env python
# coding: utf-8

# # Part1: Data Collection

# In[11]:


# 1. importing useful libraries

import requests # to get the website
import time     # to force our code to wait a little before re-trying to grab a webpage
import re       # to grab the exact element we need
from bs4 import BeautifulSoup # to grab the html elements we need
import pandas as pd


# In[12]:


movies = ['gangs_of_new_york','Atlantics','corporate_animals','playing_with_fire_2019','arctic_dogs','housefull_4','toy_story_4','honey_boy','linda_ronstadt_the_sound_of_my_voice','parasite_2019','recorder_the_marion_stokes_project','gemini_man_2019','midway_2019','zombieland_double_tap','abominable','ford_v_ferrari','doctor_sleep','maleficent_mistress_of_evil','jojo_rabbit','the_lighthouse_2019','terminator_dark_fate','after_the_wedding_2019','angel_has_fallen','everything_must_go','motherless_brooklyn','the_addams_family_2019','noelle_2019','primal','47_meters_down_uncaged','the_kitchen','in_the_tall_grass_2019','killerman','the_divine_fury','quartet_1981','pretenders','who_killed_cock_robin_2017','the_art_of_racing_in_the_rain','drive_2019','national_lampoon_s_gold_diggers','one_missed_call','ballistic_ecks_vs_sever','problem_child','return_to_the_blue_lagoon','wagons_east','3_strikes','homecoming','a_thousand_words','gotti_2017','1193743_step_brothers','the_fountain','10011582_TRON_legacy','the_secret_life_of_walter_mitty_2013','miami_vice','beerfest','hot_rod','high_tension_switchblade_romance',]

data  = []  

# access the webpage as Chrome
my_headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'}


# In[14]:


for m in movies:
    #Get page number of reviews of movie 'm'  
    pageUrl = 'https://rottentomatoes.com/m/'+ m + '/reviews'
    response = requests.get(pageUrl, headers = my_headers)
    src = response.content # content->list of bytes | response.text -> str
    # string -> html(bs4)
    soup = BeautifulSoup(src.decode('ascii', 'ignore'), 'lxml')
    # html -> data
    pageInfo = soup.find('span', {'class':re.compile('pageInfo')})
    if pageInfo==None:
        numPage=1
    else:
        numPage=int(pageInfo.text.strip().split()[-1])
    print(numPage)
    
    for k in range(1,numPage+1):
        pageUrl = 'https://rottentomatoes.com/m/'+ m + '/reviews?page='+ str(k)
        src  = False

        for i in range(1,6): 
            try:
                # get url content
                response = requests.get(pageUrl, headers = my_headers)
                # get the html content
                src = response.content
                # if we successuflly got the file, break the loop
                break 
                # if requests.get() threw an exception, i.e., the attempt to get the response failed
            except:
                print ('failed attempt #',i)
                # wait 2 secs before trying again
                time.sleep(2)


        # if we could not get the page 
        if not src:
            # couldnt get the page, print that we could not and continue to the next attempt
            print('Could not get page: ', pageUrl)
            #move on to the next page
            continue
        else:
            # got the page, let the user know
            print('Successfully got page: ', pageUrl)

        soup = BeautifulSoup(src.decode('ascii', 'ignore'), 'lxml')
        reviews = soup.findAll('div', {'class':re.compile('row review_table_row')})



        for review in reviews:

            # initialize to not found
            reviewer_name = 'NA'
            review_rating  = 'NA'
            review_source = 'NA'
            review_text = 'NA'
            review_date = 'NA'


            a = review.find('a', {'class':re.compile('unstyled bold articleLink')})
            if a:
                reviewer_name = a.text.strip()
                #print(reviewer_name)


            b = review.find('div', {'class':re.compile('review_icon icon small rotten')})
            if b:
                review_rating = 'rotten'
            else:
                review_rating = 'fresh'
                #print(review_rating)


            c = review.find('em', {'class':re.compile('subtle critic-publication')})
            if c:
                review_source = c.text.strip()
                #print(review_source)


            d = review.find('div', {'class':re.compile('the_review')})
            if d:
                review_text = d.text.strip()
                #print(review_text)


            e = review.find('div', {'class':re.compile('review-date subtle small')})
            if e:
                review_date = e.text.strip()
                #print(review_date)

            data.append([reviewer_name, review_rating, review_source, review_text, review_date])
            
with open('rotten_tomatoes_reviews.txt', mode='w', encoding='utf-8') as f:
    for review in data:
        f.write(review[0] + '\t' + review[1] + '\t' +  review[2] + '\t' +  review[3] + '\t' + review[4] + '\n')


# In[9]:


col_names=['Reviewer', 'Rating','Source', 'Content', 'Date'] 
pd.read_csv('rotten_tomatoes_reviews.txt',sep='\t',names=col_names,header=None)


# In[10]:


src = response.content


# In[11]:


print(src)


# In[14]:


print(type(src))


# In[13]:


print(src.decode('ascii', 'ignore'), 'lxml')


# In[15]:


print(type(src.decode('ascii', 'ignore')))


# with open('files/rotten_tomatoes_reviews.txt', mode='r', encoding = 'utf-8') as f:
#     data = f.read()
#     
# data = data.split('\n')[0:-1]
# 
# for i in range(0,len(data)):
#     data[i] = data[i].split('\t')
# 
# 
# print(data)

# # Part 2: Data Preprocessing

# In[7]:


import pandas as pd


# In[8]:


df=pd.DataFrame(data,columns=["Name","Rating","Source","Comment","Date"])


# In[9]:


df.shape


# In[10]:


df.head()


# In[11]:


df.to_csv('movies.csv')


# In[12]:


import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[13]:


df=pd.DataFrame(data,columns=["Name","Rating","Source","Comment","Date"])
df.to_csv('RV.csv')
df=df.dropna(subset=['Comment','Rating'])# clear all na 


# In[14]:


# map rating to binary
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid")
sns.countplot("Rating", data=df)


# In[15]:


# undersample the review data, makes data balanced
dfok=df[df['Rating']=='fresh']
dfbad=df[df['Rating']=='rotten']

ddf=pd.concat([dfok.sample(len(dfbad)),dfbad])
Xdata=ddf['Comment']
Ydata=ddf['Rating']


# In[16]:


sns.countplot("Rating", data=ddf)


# In[41]:


#nltk.download('stopwords')
sw=stopwords.words('english')


# In[49]:


from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')

#Reduce words to their root form
lemmed_sent = []
for w in sw:
    lemmed=[]
    for word in w:
        lemmed.append(WordNetLemmatizer().lemmatize(word))
    lemmed=(' ').join(lemmed)
    lemmed_sent.append(lemmed)


# In[50]:


vectorizer = TfidfVectorizer(max_features=2500, min_df=0.01, max_df=0.8)#, stop_words=stopwords.words('english'))


# In[51]:


X_processed = vectorizer.fit_transform(Xdata).toarray() #Convert text to tf-idf matrix
Y_processed=Ydata.replace('rotten',0.0).replace('fresh',1.0) #binarize


# # Part 3: Baseline Classifier

# In[59]:


X_train, X_test, Y_train, Y_test = train_test_split(X_processed,Y_processed, test_size=0.2)


# In[60]:


text_classifier = RandomForestClassifier(n_estimators=500,max_leaf_nodes=10)
text_classifier.fit(X_train, Y_train)
predictions = text_classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(Y_test,predictions))
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support
print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test, predictions))
print(text_classifier.score(X_train,Y_train))
print(text_classifier.score(X_test,Y_test))


# # Part 4: Improvement

# In[63]:


import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import *
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from subprocess import call

#import data
df=pd.read_csv("rrvv.txt",sep='{',header=None)

df=df.dropna()

#stopwords
sw=stopwords.words('english')
sw_movie=[
# 'movie',
# 'lewis',
# 'film',
# 'one',
# 'really',
# 'dicaprio',
# 'story',
# 'time',
# 'still',
# 'movies',
# 'though',
# 'made',
# 'get',
# 'way',
# 'films',
# 'thing',
# 'got',
]
sw=sw+sw_movie

Xdata=df[1]
Ydata=df[0]

vectorizer = TfidfVectorizer(max_features=2500, min_df=30, max_df=1.0, stop_words=sw)
X_processed = vectorizer.fit_transform(Xdata).toarray()
vectorizer.get_feature_names()

#convert to float because it's string originally due to some reasons
Y_processed=Ydata.astype('float64')
# define score>4 as good and score<=4 as bad
Y_processed=Y_processed.apply(lambda x: x>4)

X_train, X_test, Y_train, Y_test = train_test_split(X_processed,Y_processed, test_size=0.2)

RFclassifier = RandomForestClassifier(n_estimators=1000,max_leaf_nodes=16) 
RFclassifier.fit(X_train, Y_train)
predictions = RFclassifier.predict(X_test)
print(RFclassifier.score(X_train,Y_train))
print(RFclassifier.score(X_test,Y_test))

LRclassifier = (solver='lbfgs') 
LRclassifier.fit(X_train, Y_train)
predictions = LRclassifier.predict(X_test)
print(LRclassifier.score(X_train,Y_train))
print(LRclassifier.score(X_test,Y_test))

#output importance feature and their names
fea=vectorizer.get_feature_names()
imp=RFclassifier.feature_importances_
impdf=pd.DataFrame(imp,fea)
#export to csv with (feature name, importance)
impdf.sort_values(by=0,ascending=False).to_csv("simp.csv")


# # Part 5: EDA

# In[64]:


#output importance feature and their names
fea=vectorizer.get_feature_names()
imp=text_classifier.feature_importances_
impdf=pd.DataFrame(imp,fea)
#export to csv with (feature name, importance)
impdf.sort_values(by=0,ascending=False).to_csv("simp.csv")
#draw bar graph
impdf.sort_values(by=0,ascending=False)[:20].plot.bar()


# In[ ]:




