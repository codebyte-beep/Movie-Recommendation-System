# CS328-ProjectReport
This project aims to create a personalized movie recommendation system that offers suggestions to users based on their tastes. In order to provide personalized suggestions, the system will analyze information such as ratings and genres using collaborative filtering and content-based filtering approaches. Data has been gathered from publicly accessible movie databases, and the system will be created using machine learning methods. The project seeks to deliver an effective and tailored movie recommendation system that improves user engagement and experience.
## Introduction
A recommendation system is a type of information filtering system that predicts how a user will rate or rank a particular item. Essentially, it suggests relevant
items to users based on their interests. In the case of Netflix, which movie to watch, In the case of e-commerce, which product to buy, or In the case of Kindle, which book to read, etc.

A movie recommendation system will suggest movies to users based on their preferences. Different methods can be used to create a movie recommendation system. Collaborative filtering is a popular technique that finds people with similar tastes based on historical user behavior and suggests films those users have liked. Another technique is content-based filtering, which makes movie recommendations based on the traits of the movies the user has previously appreciated.
## Problem Statement
The aim of this project is to build a semi-realistic movie recommendation system that recommends movies to users based on their preferences and likings.
## Methodology
### Dataset
Data is taken from MovieLens. It contains 25000095 ratings across 62423 movies. Users for this dataset were selected randomly, and each of them was represented by an Id. It was collected from 162541 users between January 09, 1995, and November 21, 2019. The dataset was generated on November 21, 2019. The dataset contains two files: movies.csv and ratings.csv. Movies.csv has movieId, title, and genres as its columns, while, Rating.csv has userId, movieId, rating, and timestamp as its columns.

* MovieId: It contains the unique ids of each movie
* Title: It has the movie name and the release date of each movie
* Genres: It contains types of movies like adventure, romance, comedy, animation, etc.
* UserId: Every user has a unique id.
* Rating: Rating which the user has given to the movie from 0 to 5
### Cosine Similarity
A measure of similarity between two non-zero vectors in an inner product space is called cosine similarity. In order to offer related items to clients, recommendation systems frequently employ cosine similarity. For instance, if a consumer buys a certain book, the recommendation system can utilize cosine similarity to discover additional books with the same subject matter and suggest them to the customer. In our case, we use it to find similarities between two movies that are represented by vectors.

The cosine of the angle between two vectors is measured using cosine similarity. If the vectors are equal, their cosine similarity will be 1, and their angle will be 0. The cosine similarity is zero, and the angle between the vectors is 90 degrees if they are orthogonal (perpendicular). The cosine similarity is -1, and the angle between the vectors is 180 degrees if they are moving in opposite directions. The fact that cosine similarity is computationally quick and effective with high-dimensional data is one of its advantages. This makes it a popular option for machine learning and large-scale data processing applications where efficiency is crucial.
### Dataset Preprocessing
Before moving towards building the recommendation system, we have first to preprocess the data to remove the unwanted stuff and create useful data. 

* There are genres in the movie dataset that are listed as empty. So, we first dropped all the movies for which no genres were listed from the file movies. It is done by finding the index of the movie where genres are not given, and after getting the index, use the drop option to remove the movie from that index. 
* The movie titles had years of release also written along with the name. This would have created a problem while searching for recommended movies by giving movie titles, as one might not remember the year of release of his favorite movie. So we separated the release year from the titles and created a new column for the released year for each movie.
* There were some unwanted symbols like ’|’ in the ‘genres’ column. These symbols would have created problems while separating the genres of the movies for making content-based recommendations. So, we have replaced these with blank spaces.
* In the rating file, we dropped the column ‘timestamp’, which was unwanted.
* We filtered out only those users from the dataset who have rated more than 1000 movies. We also filtered out only those movies which have been rated by at least 50 users. This is done with the intent to give better movie suggestions by just keeping experienced and popular movies into consideration.
* We finally replaced all the null values in the similarity matrix with 0.
## Preparing a Content-Based Recommender system
In the content-based movie recommendation system, a movie is provided based on the watch history of the viewer, genres, and tags are used to find similarities
between the movies, and then the most similar movies are recommended. We have used the genres as tags to prepare the content-based system, and it recommends movies that are similar to each other based on their genres.

We used the CountVectorizer function from the Scikit learn(sklearn) library to create a sparse matrix for the text having columns of unique words. Each movie has genres listed, so it will create a row having entries for each genres items. We are basically converting the genre terms into arrays. These arrays would finally be used for comparing the movies based on their genres.

Thereafter, to find the similarity between the genre arrays created for movies, we used the cosine similarity function from the sklearn library. We take a movie and find out its similarity scores of it with other all the other movies on the basis of their genres. 

For the recommendation function part, we select a movie and get the index of that movie. We use this index to find out the similarity vector of that movie with other movies. Finally, we sort the list of similarity scores for a movie and select only the top 20 movies based on the similarity scores. We recommend these movies to the user through their titles listed.
## Moving toward Collaborative-Filtering based Recommendation System
The dataset which we used only had the genres listed for the movies. These genres were very few in number to be kept as tags for recommending movies based on content. As a result, the movies recommended by the content-based mechanism weren’t much satisfying. Therefore, for enhancing our results of recommended movies we move ahead toward the collaborative filtering mechanism. Collaborative filtering mechanism works by finding similar patterns or information of the users, this technique can filter out items that users like on the basis of the ratings or reactions by similar users.

For building the collaborative filtering-based recommender function, we first merge the movie and rating dataset using the key as movieId. So that we have a new ’movierating’ dataset having movie name with their users’ ratings. After that we group together the data with the same ’userId’ and store only the data of those users who have rated more than 1000 movies.Also, we grouped together the data with the same ’movieId’ and stored only those movies which have been rated by at least 50 users.Finally we had a dataset that had users who have rated more than 1000 movies, and only those movies which have been rated by at least 50 users.

We then create a matrix with the movie title as rows, userId as columns, and the values as the ratings given to the movie by the corresponding users. We got a shape of 9770 × 2665. The null values in this matrix signified the places where the users have not given any ratings to that particular movie. We replaced these null values with 0.

Thereafter, we again use the cosine similarity function from the Scikit library(sklearn) to find similarity scores for each movie with respect to other movies based on the user ratings. These similarity scores tell us how closer a movie is to another movie based on the user ratings. It contains numerical values between 0 to 1. 

In the recommendation part, we select a movie and find out its index from the rating matrix. After getting the index, we take out the similarity scores for that index, and we get an array, which we finally sort in descending order so that we can select the top 20 movies based on their similarity scores. We list them all in a list named ’similaritems’, and we get the title for each of these movies present in the list through the rating matrix. We finally print the titles of all the recommended movies in the list format.
## Enhancing the Collaborative Filtering mechanism
The collaborative filtering mechanism definitely gives better results of recommended movies when compared with the contest-based recommendation
mechanism. However, in the collaborative filtering mechanism, the movies were recommended only based on the user rating similarities, i.e. the movies which were being recommended did not have many similarities between them. 

To overcome this problem and to recommend movies that are similar to each other as well, we merged the concepts of the content-based with those of the collaborative filtering mechanism. We added a function that, while recommending movies based on user ratings, was also considering the genres of the movies so that the similarity between the movies could also be maintained. This enhanced collaborative filtering function was further giving better results in terms of movie recommendations when compared with the simple collaborative filtering mechanism.
## Results
We analyze our results based on the ratings that the user gives to the movies which were recommended to him based on his particular selection of a movie.

Based on these results, the enhanced collaborative filtering mechanism recommended better movies as it was considering the concept of both the content based as well as collaborative filtering-based mechanisms.
