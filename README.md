# ADBI-projects
Homework and projects done in the course CSC591 Algorithms for Data-Guided Business Intelligence (ADBI)  (Spring 2019) <br> 
**Instructor: Prof. Nagiza Samatova**

## Projects
### Project 1: Data Wrangling with Python and Pandas
Concepts:
- Data Acquisition
- Data Cleansing
- Data Understanding: Basics
- Data Manipulation
### Project 2: Recommender System with ALS and Apache Spark
Create a recommender system using Spark and the collaborative filtering technique that will recommend new musical artists to a user based on their listening history.
### Project 3: Sentiment Analysis with Spark Streaming
Perform a basic sentiment analysis of realtime tweets, i.e., processing live data streams using Spark’s streaming APIs and Python.
### Project 4: Network Properties with Apache Spark
Implement various network properties using pySpark, GraphFrames and networkx:
- Degree Distribution: a measure of the frequency of nodes that have a certain degree
- Centrality: determine nodes that are important based on the structure of the graph. Closeness centrality measures the distance of a node to all other nodes.
- Articulation Points: vertices in the graph that, when removed, create more components than there were originally. 
### Project 5: Bitcoin Price Prediction with Bayesian Regression
predicting the price variations of bitcoin, a virtual cryptographic currency using Bayesian Regression. Using [this](http://arxiv.org/pdf/1410.1231.pdf) paper as reference computer price variations, linear regression parameters and build linear regression model with bayesian estimates.
### Project 6: AdWords Placement with Online Bipartite Matching
We are given a set of advertisers each of whom has a daily budget 𝐵𝑖. When a user
advertisement slot. The bid of advertiser 𝑖 for an ad request 𝑞 is denoted as 𝑏 . We assume that 𝑖𝑞
performs a query, an ad request is placed online and a group of advertisers can then bid for that
the bids are small with respect to the daily budgets of the advertisers (i.e., for each 𝑖 and 𝑞, 𝑏𝑖𝑞 ≪ 𝐵𝑖). Moreover, each advertisement slot can be allocated to at most one advertiser and the advertiser is charged his bid from his/her budget. The objective is to maximize the amount of money received from the advertisers.
For this project, we make the following simplifying assumptions:
1. For the optimal matching (used for calculating the competitive ratio), we will assume
everyone’s budget is completely used. (optimal revenue = the sum of budgets of all
advertisers)
2. The bid values are fixed (unlike in the real world where advertisers normally compete by
incrementing their bid by 1 cent).
3. Each ad request has just one advertisement slot to display.
### Project 7: Market Segmentation & Influence Propagation 
Market segmentation divides a broad target market into subsets of consumers or businesses that have or are perceived to have common needs, interests, and priorities. In this project, we aim to find such market segments given social network data. These social relations can be captured in a graph framework where nodes represent customers/users and edges represent some social relationship. The properties belonging to each customer/user can be treated as node attributes. Hence, market segmentation becomes the problem of community detection over attributed graphs, where the communities are formed based on graph structure as well as attribute similarities. We evaluate the obtained segments via influence propagation (influence an entity in each segment and measure how fast the influence propagates over the entire network).
### Project 8: DeepWalk: Multipartite Graph Embedding for Recommender Systems
Predict a user’s preference for some item they have not yet rated using a collaborative filtering graph-based technique called DeepWalk. The main steps are:
1. Create a heterogeneous information network with nodes consisting of users, item- ratings, items, and other entities related to those items
2. Use DeepWalk to generate random walks over this graph
3. Based on these random walks, embed the graph in a low dimensional space using
word2vec.
Evaluate and compare preference propagation algorithms in heterogeneous information networks generated from user-item relationships. Implement and evaluate a word2vec-based method.
### Project 9: Word2Vec and Doc2Vec: Sentiment Analysis of Text
Perform sentiment analysis over IMDB movie reviews and Twitter data to classify tweets or movie reviews as either positive or negative given a labeled training data to build the model and labeled testing data to evaluate the model. Generate embedding/feature vectors using Word2Vec and Doc2Vec techniques and build classifiers using logistic regression as well as a Naive Bayes classifier. 
### Project 10: Deep Neural Network Architectures: Defect segmentation on Textured Surfaces
Develop a model to detect defects by Industrial Optical Inspection on Textured Surfaces. The problem can be modelled as an Image Segmentation task where we can find the pixels where the defect occurs in a given image. <br>
The dataset used: https://hci.iwr.uni-heidelberg.de/node/3616 <br>
We solve this using deep learning and the approach we will follow is based on the paper on [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

### Project 11: Topic Modeling with LDA
Use LDA(Latent Dirichlet Allocation) for topic modeling which is automatic organization and summarization of large electronic unstructured text corpus.It is used to uncover the major themes (topics) that pervade the corpus.


## Homeworks
### Homework 1: GLM Logistic Regression
Build the logistic regression model (fit.all) using all the predictor in R and answer questions related to basics of logistic regression like equations, log-odds, statisitical significance, dispersion, etc.
### Homework 2: Bayesian Parameter Estimation
Bayesian Estimation of the Parameters of a Gaussian Distribution and answer related questions.
### Homework 3: Stochastic Gradient Descent for Logistic Regression
Implement SGD for the Logistic Regression problem from scratch. 

