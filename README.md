# Book_read_and_cat_prediction
CSE158B Assignment1

1. Read prediction:
In this prediction, I used the logistic regression model. I also changed the ratio of training and testing dataset to 180000:20000. There are four features included in my model. 
For each data tuple "user" and "book" I am trying to predict, 
    1. I calculate the maximum of Jaccard similarity between users who read "book" and users who read b', where b' belongs to "user" (this is similar to the hw, but in logistic regression model I just put that similarity instead of comparing it with one threshold). If there is no such user or b', the maximum is represented by 0.
    2. I also add the maximum of Jaccard similarity between books who are read by "user" and books who are read by u', where u' also read "book". If there is no such u' or book, the maximum is represented by 0.
    3. I add the number of times the book occurred in the training dataset as third feature to represent the popularity of the book.
    4. 1 as intercept.
By using the above features, I fit them into logistic regression and using a for loop to find the best parameter C, which turns out to be 0.0001.


2. Category prediction:
In this prediction, I tried to use logistic regression to classify different categories but the accuracy did not excess 0.75. So I searched online and used the lightgbm classification model instead. In my model, I first split the training set into two parts with ratio of 8:2 and get the train and validation dataset. Then I collect the first 5000 popular words excluding stop words for reviews in each category, and my model contains 8227 features in total. Then I tuned the parameter in the lightgbm, which are num_leaves = 90, n_estimators = 1000, learning_rate = 0.07, reg_alpha = 1 and reg_lambda = 0.1. 