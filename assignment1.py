#!/usr/bin/env python
# coding: utf-8

# In[188]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from nltk.corpus import stopwords
from tqdm import tqdm
import lightgbm as ltb
from sklearn.model_selection import train_test_split


# In[189]:


def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)


# In[190]:


def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')


# In[191]:


def accuracy(pred, y):
    return sum([x==y for x,y in zip(pred, y)]) / len(pred)
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[192]:


allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)


# ## Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

# In[239]:


# ratingsTrain = allRatings[:190000]
# ratingsValid = allRatings[190000:]
ratingsTrain = allRatings[:180000]
ratingsValid = allRatings[180000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
len(allRatings)


# In[240]:


bookCount = defaultdict(int)
totalRead = 0


# In[241]:


for user,book,_ in readCSV("train_Interactions.csv.gz"):
  bookCount[book] += 1
  totalRead += 1


# In[242]:


mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()


# In[243]:


# calculate most popular books
return1 = set()
count = 0
for ic, i in mostPopular:
  count += ic
  return1.add(i)
  if count > totalRead/2: break


# In[244]:


totalbook = set()
data_ub = []
booksPerUser = defaultdict(set)

for u,b,r in ratingsTrain:
    totalbook.add(b)
    booksPerUser[u].add(b)
    
for u,b,r in ratingsValid:
    totalbook.add(b)
    data_ub.append((u,b))
    booksPerUser[u].add(b)


# In[245]:


validation_0 = []
for i in data_ub:
    u = i[0]
    validation_0.append((u, random.sample(totalbook.difference(booksPerUser[u]), 1)))


# In[246]:


# 1 represents read the book
validation_set = []
for i in data_ub:
    validation_set.append([i[0], i[1], 1])
for j in validation_0:
    validation_set.append([j[0], j[1][0], 0])


# In[247]:


usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rateduser
for u,b,r in ratingsTrain:
    usersPerItem[b].add(u)
    itemsPerUser[u].add(b)


# In[272]:


book_count = {}
for ic, i in mostPopular:
    book_count[i] = ic
avg_count = numpy.mean([i[0] for i in mostPopular])
def feature(u,b):
    Jaccard_u = []
    for b2 in itemsPerUser[u]:
        if b2 == b:
            continue
        Jaccard_u.append(Jaccard(usersPerItem[b2], usersPerItem[b]))
    if Jaccard_u == []:
        tocompare = 0
    else: tocompare = numpy.max(Jaccard_u)

    Jaccard_u1 = []
    for u2 in usersPerItem[b]:
        if u2 == u:
            continue
        Jaccard_u1.append(Jaccard(itemsPerUser[u2], itemsPerUser[u]))
    if Jaccard_u1 == []:
        tocompare1 = 0
    else: tocompare1 = numpy.max(Jaccard_u1)

    book_pop = avg_count
    if b in book_count:
        book_pop = book_count[b]
    return [1, book_pop, tocompare, tocompare1]

X = [feature(i[0],i[1]) for i in validation_set]
y = [i[2] for i in validation_set]


# In[273]:


best_acc = 0
best_model = 0
for c in tqdm(range(-10, 10)):
    acc_avg = []
    acc_max_model = 0
    acc_max = 0
    for i in range(0,10):
        model = linear_model.LogisticRegression(C = 10 ** c, fit_intercept=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc = accuracy(prediction, y_test)
        acc_avg.append(acc)
        if acc > acc_max:
            acc_max = acc
            acc_max_model = model
    acc_avg = numpy.average(numpy.array(acc_avg))
    print(acc_avg, c)
    if acc_avg > best_acc:
        best_model = acc_max_model
        best_acc = acc_avg
print(best_acc, best_model.coef_)


# In[274]:


best_acc


# ##### predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         #header
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     x = feature(u,b)
#     y = best_model.predict([x])
#     predictions.write(u + ',' + b + ',' + str(y[0]) + "\n")
# predictions.close()

# In[ ]:


# def best_itemsim(threshold):
#     prediction = []
#     y = []
#     for j in validation_set:
#         Jaccard_u = []
#         u, b, status = j[0], j[1], j[2]
#         for b2 in itemsPerUser[u]:
#             Jaccard_u.append(Jaccard(usersPerItem[b2], usersPerItem[b]))
#         if Jaccard_u == []:
#             tocompare = 0
#         else: tocompare = max(Jaccard_u)
#         prediction.append(tocompare > threshold)
#         y.append(status)
#     return prediction, y


# In[ ]:


# threshold_best_itemsim = 0
# acc_itemsim = 0
# for threshold_itemsim in numpy.arange(0.002, 0.004, 0.0001):
#     prediction, y = best_itemsim(threshold_itemsim)
#     print(accuracy(prediction, y), threshold_itemsim)
#     if accuracy(prediction, y) > acc_itemsim:
#         acc_itemsim = accuracy(prediction, y)
#         threshold_best_itemsim = threshold_itemsim


# In[ ]:


# def best_usersim(threshold):
#     prediction = []
#     y = []
#     for j in validation_set:
#         Jaccard_u = []
#         u, b, status = j[0], j[1], j[2]
#         for u2 in usersPerItem[b]:
#             Jaccard_u.append(Jaccard(itemsPerUser[u2], itemsPerUser[u]))
#         if Jaccard_u == []:
#             tocompare = 0
#         else: tocompare = max(Jaccard_u)
#         prediction.append(tocompare > threshold)
#         y.append(status)
#     return prediction, y


# In[ ]:


# threshold_best_usersim = 0
# acc_usersim = 0
# for threshold_usersim in numpy.arange(0.03, 0.05, 0.001):
#     prediction, y = best_usersim(threshold_usersim)
#     print(accuracy(prediction, y), threshold_usersim)
#     if accuracy(prediction, y) > acc_usersim:
#         acc_usersim = accuracy(prediction, y)
#         threshold_best_usersim = threshold_usersim


# In[ ]:


# def best_popular(threshold):
#     prediction = []
#     y = []
#     mostPopular = [(bookCount[x], x) for x in bookCount]
#     mostPopular.sort()
#     mostPopular.reverse()
#     return1 = set()
#     count = 0
#     for ic, i in mostPopular:
#         count += ic
#         return1.add(i)
#         if count > int(totalRead * threshold): break
#     y = [i[2] for i in validation_set]
#     prediction = [1 if i[1] in return1 else 0 for i in validation_set]
#     return prediction, y


# In[ ]:


# acc_popularbook = 0
# threshold_popularbook = 0
# popularset = set()
# for threshold1 in numpy.arange(0.6, 0.8, 0.001):
#     prediction, y = best_popular(threshold1)
#     acc = accuracy(y, prediction)
#     print(acc, threshold1)
#     if acc > acc_popularbook:
#         acc_popularbook = acc
#         threshold_popularbook = threshold1
#         popularset = return1


# In[ ]:


# best_param = ()
# best_acc = 0
# for threshold1 in numpy.arange(0.774, 0.775, 0.0001):
#     prediction1, y1 = best_popular(threshold1)
#     for threshold_itemsim in numpy.arange(0.0013, 0.0015, 0.0001):
#         prediction2, y2 = best_itemsim(threshold_itemsim)
#         for threshold_usersim in numpy.arange(0.00, 0.002, 0.001):
#             prediction3, y3 = best_usersim(threshold_usersim)
#             prediction = [all((x,y,z)) for x, y, z in zip(prediction1,prediction2,prediction3)]
#             acc = accuracy(prediction, y1)
#             print(acc, best_param, (threshold1, threshold_itemsim, threshold_usersim))
#             if acc > best_acc:
#                 best_param = (threshold1, threshold_itemsim, threshold_usersim)
#                 best_acc = acc


# In[ ]:


# best_param = ()
# best_acc = 0
# prediction1, y1 = best_popular(0.77)
# prediction2, y2 = best_itemsim(0.001)
# for threshold_usersim in numpy.arange(0.000, 0.001, 0.0001):
#             prediction3, y3 = best_usersim(threshold_usersim)
#             prediction = [all((x,y,z)) for x, y, z in zip(prediction1,prediction2,prediction3)]
#             acc = accuracy(prediction, y1)
#             print(acc, best_param, (threshold1, threshold_itemsim, threshold_usersim))
#             if acc > best_acc:
#                 best_param = (threshold1, threshold_itemsim, threshold_usersim)
#                 best_acc = acc


# In[ ]:


# best_param = (0.77, 0.001, 0.0)


# In[ ]:


# def CosineSet(s1, s2):
#     # Not a proper implementation, operates on sets so correct for interactions only
#     numer = len(s1.intersection(s2))
#     denom = math.sqrt(len(s1)) * math.sqrt(len(s2))
#     if denom == 0:
#         return 0
#     return numer / denom


# In[ ]:


# threshold_best = 0
# acc4 = 0
# y = [i[2] for i in validation_set]
# for threshold4_1 in numpy.arange(0.7, 0.8, 0.01):
#     return2 = set()
#     count = 0
#     for ic, i in mostPopular:
#         count += ic
#         return2.add(i)
#         if count > int(totalRead * threshold1): break
#     for threshold4_2 in numpy.arange(0.002, 0.003, 0.0001):
#         prediction = []
#         for j in validation_set:
#             Jaccard_u = []
#             u, b, status = j[0], j[1], j[2]
#             for b2 in itemsPerUser[u]:
#                 Jaccard_u.append(Jaccard(usersPerItem[b2], usersPerItem[b]))
#             if Jaccard_u == []:
#                 tocompare = 0
#             else: tocompare = max(Jaccard_u)
#             prediction.append(tocompare > threshold4_2 and b in return2)
#         if accuracy(prediction, y) > acc4:
#             acc4 = accuracy(prediction, y)
#             threshold_best = (threshold4_1, threshold4_2)
#         print(threshold4_1, threshold4_2)


# In[ ]:


# popularset = set()
# count = 0
# for ic, i in mostPopular:
#     count += ic
#     popularset.add(i)
#     if count > int(totalRead * best_param[0]): break
# threshold_best_itemsim = best_param[1]
# threshold_best_usersim = best_param[2]


# In[ ]:


# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         #header
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     Jaccard_u = []
#     for b2 in itemsPerUser[u]:
#         Jaccard_u.append(Jaccard(usersPerItem[b2], usersPerItem[b]))
#         if Jaccard_u == []:
#             tocompare1 = 0
#         else: tocompare1 = max(Jaccard_u)
#     Jaccard_i = []
#     for u2 in usersPerItem[b]:
#         Jaccard_i.append(Jaccard(itemsPerUser[u2], itemsPerUser[u]))
#         if Jaccard_i == []:
#             tocompare_2 = 0
#         else: tocompare_2 = max(Jaccard_i)
#     if b in popularset and tocompare1 > threshold_best_itemsim and tocompare_2 > threshold_best_usersim:
#         predictions.write(u + ',' + b + ",1\n")
#     else:
#         predictions.write(u + ',' + b + ",0\n")


# In[ ]:





# In[ ]:


# predictions = open("predictions_Read.csv", 'w')
# for l in open("pairs_Read.csv"):
#     if l.startswith("userID"):
#         predictions.write(l)
#         continue
#     u,b = l.strip().split(',')
#     Jaccard_u = []
#     for b2 in itemsPerUser[u]:
#         Jaccard_u.append(Jaccard(usersPerItem[b2], usersPerItem[b]))
#         if Jaccard_u == []:
#             tocompare = 0
#         else: tocompare = max(Jaccard_u)
#     print(l[:-1] + "," + str(int(tocompare > threshold_best_itemsim and b in popularset)) + "\n")
#     predictions.write(l[:-1] + "," + str(int(tocompare > threshold_best_itemsim and b in popularset)) + "\n")


# ## Category prediction baseline: Just consider some of the most common words from each category

# In[222]:


catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}


# In[223]:


data = []

for d in readGz("train_Category.json.gz"):
    data.append(d)


# hw3 + distinctive feature for each genre

# In[276]:


data[0]


# In[277]:


from sklearn.feature_extraction.text import CountVectorizer


# In[278]:


punctuation = set(string.punctuation)
dfs = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
for d in data:
    df = dfs[d['genreID']]
    r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
    for w in set(r.split()):
        df[w] += 1
for i in stopwords.words('english'):
    for df in dfs:
        if i in df:
            del df[i]
counts = [sorted([(v,k) for k, v in df.items()], reverse=True) for df in dfs]


# In[279]:


len(counts)


# In[280]:


words_cat = [set()] * 5
for idx in tqdm(range(5)):
    for i in counts[idx][:5000]:
        words_cat[idx].add(i[1])


# In[281]:


all_five_cat_words = words_cat[0]
for i in words_cat:
    all_five_cat_words = all_five_cat_words.union(i)


# In[282]:


wordId = dict(zip(all_five_cat_words, range(len(all_five_cat_words))))
wordSet = set(all_five_cat_words)


# In[283]:


len(wordSet)


# In[285]:


all_five_cat_words


# In[97]:


vectorizer = CountVectorizer()
vectorizer.fit_transform(all_five_cat_words)


# In[99]:


vectorizer.transform([data[0]['review_text']])


# In[286]:


def feature_vector(datum):
    feat = vectorizer.transform([datum['review_text']]).toarray()
    return feat[0]
X = [feature_vector(d) for d in data]
y = [d['genreID'] for d in data]


# In[287]:


Xtrain = numpy.array(X[:8*len(X)//10])
ytrain = numpy.array(y[:8*len(y)//10])
Xvalid = numpy.array(X[8*len(X)//10:])
yvalid = numpy.array(y[8*len(y)//10:])


# In[288]:


#vectorize lgbm
mod = ltb.LGBMClassifier(num_leaves = 100, n_estimators = 1000, learning_rate = 0.07, reg_alpha = 1, reg_lambda = 0.1, n_jobs = -1,objective = 'multiclass')
mod.fit(Xtrain, ytrain)
pred = mod.predict(Xvalid)
correct = pred == yvalid
print(sum(correct) / len(correct))


# In[289]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    
    x = feature_vector(test_data[pos])
    
#     prediction = best_model.predict([x])
    prediction = mod.predict([x])
    print(l[:-1] + "," + str(prediction[0]) + "\n")
    predictions.write(l[:-1] + "," + str(prediction[0]) + "\n")
    pos += 1
predictions.close()


# In[123]:



#vectorize lgbm
# mod = ltb.LGBMClassifier(num_leaves = 100, n_estimators = 1000, learning_rate = 0.07, reg_alpha = 1, reg_lambda = 0.1, n_jobs = -1,objective = 'multiclass')
# mod.fit(Xtrain, ytrain)
# pred = mod.predict(Xvalid)
# correct = pred == yvalid
# print(sum(correct) / len(correct))


# In[121]:


#vectorize logisticregression
best_model = 0
best_acc = 0
for c in tqdm(range(-2, 4)):
    mod = linear_model.LogisticRegression(C=10 ** c)
    mod.fit(Xtrain, ytrain)

    pred = mod.predict(Xvalid)
    correct = pred == yvalid
    if sum(correct) / len(correct) > best_acc:
        best_model = mod
        best_acc = sum(correct) / len(correct)
    print(c, sum(correct) / len(correct))


# In[76]:


# lgbm
best_model = 0
best_acc = 0
for n in tqdm([60, 70, 80, 90]):
    mod = ltb.LGBMClassifier(num_leaves = n, n_estimators = 1000, learning_rate = 0.07, reg_alpha = 1, reg_lambda = 0.1, n_jobs = -1)
    mod.fit(Xtrain, ytrain)
    pred = mod.predict(Xvalid)
    correct = pred == yvalid
    if sum(correct) / len(correct) > best_acc:
        best_model = mod
        best_acc = sum(correct) / len(correct)
    print(n, sum(correct) / len(correct))


# In[41]:


# n_estimator: 1000, learning_rate:0.04 0.7395
# n_estimator: 1000, learning_rate: 0.07, reg_alpha: 1, reg_lambda = 0.1, acc: 0.7507
# :3000 len(feature) = 4867

# :2000 acc: 0.7303 numleaves = 50


# In[ ]:





# In[21]:


# no stem
best_model = 0
best_acc = 0
for c in tqdm(range(-2, 4)):
    mod = linear_model.LogisticRegression(C=10 ** c)
    mod.fit(Xtrain, ytrain)

    pred = mod.predict(Xvalid)
    correct = pred == yvalid
    if sum(correct) / len(correct) > best_acc:
        best_model = mod
        best_acc = sum(correct) / len(correct)
    print(c, sum(correct) / len(correct))


# In[ ]:



best_model = 0
best_acc = 0
for c in tqdm(range(-2, 4)):
    mod = linear_model.LogisticRegression(C=10 ** c)
    mod.fit(Xtrain, ytrain)

    pred = mod.predict(Xvalid)
    correct = pred == yvalid
    if sum(correct) / len(correct) > best_acc:
        best_model = mod
        best_acc = sum(correct) / len(correct)
    print(c, sum(correct) / len(correct))


# In[114]:


#每个category前5000，gs acc0.735， regress C=0.1，
best_model, best_acc


# In[56]:


test_data = []
for l in readGz('test_Category.json.gz'):
    test_data.append(l)


# In[59]:


predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    
    x = feature(test_data[pos])
    
#     prediction = best_model.predict([x])
    prediction = mod.predict([x])
    print(l[:-1] + "," + str(prediction[0]) + "\n")
    predictions.write(l[:-1] + "," + str(prediction[0]) + "\n")
    pos += 1
predictions.close()


# In[ ]:


catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}


# In[79]:


person_each_category = [[], [], [], [], []]
for i in data:
    
    person_each_category[i['genreID']].append(i)


# In[85]:


for i in person_each_category:
    print(numpy.mean(numpy.array([len(j['review_text']) for j in i])))


# In[94]:





# smallds_test

# In[55]:


words_cat = [set()] * 5
for idx in tqdm(range(5)):
    for i in counts[idx][:1000]:
        words_cat[idx].add(i[1])


# In[56]:


all_five_cat_words = words_cat[0]
for i in words_cat:
    all_five_cat_words = all_five_cat_words.union(i)


# In[57]:


len(all_five_cat_words)


# In[58]:


wordId = dict(zip(all_five_cat_words, range(len(all_five_cat_words))))
wordSet = set(all_five_cat_words)


# In[66]:


def feature(datum):
    feat = [0]*len(wordSet)
    r = ''.join([c for c in datum['review_text'].lower() if not c in punctuation])
    length = len(r.split())
    for w in r.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    feat.append(1) #offset
    feat.append(length)
    return feat

X = [feature(d) for d in data[:5000]]
y = [d['genreID'] for d in data[:5000]]


# In[68]:


Xtrain = X[:9*len(X)//10]
ytrain = y[:9*len(y)//10]
Xvalid = X[9*len(X)//10:]
yvalid = y[9*len(y)//10:]


# In[70]:


best_model = 0
best_acc = 0
for c in tqdm(range(0, 12)):
    mod = linear_model.LogisticRegression(C=10 ** c)
    mod.fit(Xtrain, ytrain)

    pred = mod.predict(Xvalid)
    correct = pred == yvalid
    if sum(correct) / len(correct) > best_acc:
        best_model = mod
        best_acc = sum(correct) / len(correct)
    print(c, sum(correct) / len(correct))


# In[62]:


# occ/ length
best_acc


# In[67]:


# no length
best_acc


# In[71]:


# length
best_acc


# In[ ]:





# In[111]:





# pipeline

# In[25]:


def feature(datum, words, wordId, tolower=True, removePunct=True):
    feat = [0]*len(words)
    r = datum['review_text']
    if tolower:
        r = r.lower()
    if removePunct:
        r = ''.join([c for c in r if not c in punctuation])
    for w in r.split():
        if w in words:
            feat[wordId[w]] += 1
    feat.append(1) # offset
    return feat


# In[26]:


def pipeline(dSize = 5000, tolower=True, removePunct=True, ratio = 0.8, dratio = 1.0):
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in data: # Strictly, should just use the *training* data to extract word counts
        r = d['review_text']
        if tolower:
            r = r.lower()
        if removePunct:
            r = ''.join([c for c in r if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()
    
    words = [x[1] for x in counts[:dSize]]
    
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)
    
    data_sub = data[:int(dratio * len(data))]
    X = [feature(d, words, wordId, tolower, removePunct) for d in data_sub]
    y = [d['genreID'] for d in data_sub]
    
    Ntrain,Nvalid,Ntest = int(len(X) * ratio), len(X) - int(len(X) * ratio), 0
    Xtrain,Xvalid,Xtest = X[:Ntrain],X[Ntrain:Ntrain+Nvalid],X[Ntrain+Nvalid:]
    ytrain,yvalid,ytest = y[:Ntrain],y[Ntrain:Ntrain+Nvalid],y[Ntrain+Nvalid:]
    
    bestModel = None
    bestVal = None
    bestLamb = None
    
    ls = list(range(3, 5))
    errorTrain = []
    errorValid = []

    for l in tqdm(ls):
        print(l)
        model = linear_model.LogisticRegression(C = 10 ** l)
        model.fit(Xtrain, ytrain)
        predictTrain = model.predict(Xtrain)
        acctrain = accuracy(predictTrain, ytrain)
        errorTrain.append(acctrain)
        predictValid = model.predict(Xvalid)
        accvalid = accuracy(predictValid, yvalid)
        print("l = " + str(l) + ", acc= " + str(accvalid))
        if bestVal == None or accvalid > bestVal:
            bestVal = accvalid
            bestModel = model
            bestLamb = l
    return bestVal, bestModel, bestLamb


# In[27]:


outcome = pipeline()


# In[28]:


outcome[1]


# In[31]:


wordCount = defaultdict(int)
punctuation = set(string.punctuation)
for d in data: # Strictly, should just use the *training* data to extract word counts
    r = d['review_text']
    r = r.lower()
    r = ''.join([c for c in r if not c in punctuation])
    for w in r.split():
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()
words = [x[1] for x in counts[:5000]]

wordId = dict(zip(words, range(len(words))))


# In[32]:



predictions = open("predictions_Category.csv", 'w')
pos = 0

for l in open("pairs_Category.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    
    x = feature(test_data[pos], words, wordId)
    
    prediction = outcome[1].predict([x])
    print(l[:-1] + "," + str(prediction[0]) + "\n")
    predictions.write(l[:-1] + "," + str(prediction[0]) + "\n")
    pos += 1
predictions.close()


# In[34]:


X[0]


# In[ ]:




