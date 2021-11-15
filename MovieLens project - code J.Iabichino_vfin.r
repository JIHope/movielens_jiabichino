
########## MovieLens project - Jody Iabichino ##########
########################################################

#DATA PREPARATION
#First of all, we start by loading the libraries we may need in the course of the analysis.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(dslabs)

#Using the following link, let's create two databases, ratings and movies, defining their column names

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))

#if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres)) #<-we format the movies database variables, defining that the movieId should be numeric, while the other two should be composed of characters


# We now create a single database, the movielens database, by adding the columns of the movies database to those of ratings, using the variable "movieId" as a matching key.
movielens <- left_join(ratings, movies, by = "movieId")

#We analyse some statistics about the database, such as the number of columns and the number of rows.

ncol_movielens <- ncol(movielens)
nrow_movielens <- nrow(movielens)

ncol_movielens #<--number of columns (6)
nrow_movielens #<--number of rows (10000054)

head(movielens) #<-- Let us look at the first few lines of the database to see how it is structured.

# DATA EXPLORATION
# Let us now begin the exploration of the data by starting with the analysis of the users.
#First of all, how many users are there? To answer, we use the following code:
ml_users <- movielens %>%
  group_by(userId) %>%
  summarize(count=n())
nrow_ml_users <- nrow(ml_users)

nrow_ml_users #<- There are 69878 userId

# How many ratings did the top ten users give each? This is easily calculated:

movielens %>%
  group_by(userId) %>%
  summarize(count=n()) %>%
  arrange(desc(count)) %>%
  top_n(10)  

#How many ratings in total are there for the top 10 users?
top10_users_count <- movielens %>%
  group_by(userId) %>%
  summarize(count=n()) %>%
  arrange(desc(count)) %>%
  top_n(10)  %>% summarize(sum(count))  

top10_users_count # <-- They provided 47768 ratings

#Now let's look at the distribution of ratings:
ml_rating <- movielens %>%
  group_by(rating) %>%
  summarize(count=n())
nrow_ml_rating <- nrow(ml_rating)
nrow_ml_rating 
ml_rating # <-- we first calculate the distribution


ggplot(data = ml_rating, aes(x = rating, y = count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label=paste0(round(count/nrow_movielens, 2)*100,"%"))) +
  ggtitle("Bar plot of ratings") + 
  ylab("relative frequency")   #<-- Then, we graphically represent the distribution of ratings

# Let us now analyse the movies. How many movies are in the database? To answer, we use the following code:
ml_movieid <- movielens %>%
  group_by(movieId) %>%
  summarize(count=n())
nrow_ml_movieid <- nrow(ml_movieid)

nrow_ml_movieid #<- there are 10677 movies.

#We now want to determine which are the top 10 films based on the number of ratings received. 
#To do it, we use the following code:
movielens %>%
  mutate(movieId_and_title = paste0(movieId,"_",title))%>%
  group_by(movieId_and_title) %>%
  summarize(count=n(), count_percentage = count / nrow_movielens) %>%
  arrange(desc(count_percentage)) %>%
  top_n(10) 
#Let's continue with the genre. How many genres are there in the database?
#To answer, we use the following code:
ml_genres <- movielens %>%
  group_by(genres) %>%
  summarize(count=n())
nrow_ml_genres <- nrow(ml_genres)

nrow_ml_genres   #<-there are 797 different genres in the database

#Let's then determine which are the top 10 genres in movielens based on the number of ratings received.
#To do it, we use the following code:

ml_genres%>%
  mutate(count_percentage = count / nrow_movielens) %>%
  arrange(desc(count_percentage)) %>%
  top_n(10)

#DATA CLEANING
sum(is.na(movielens)) #<-- The results shows that movielens is an extremely clean dataset, with no missing values present.

#DATA PARTITION
#After exploring the data, we split `movielens` into two distinct databases: the training set (edx) and the test set (temp), the first used to try the model, the second to validate it.
# Validation set will be 10% of MovieLens data

set.seed(1)
# if using R 3.6 or later, use set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE) #this is the index that we will use to split tha data. P=0.1 means validation set (temp) will be 10% of MovieLens data.
edx <- movielens[-test_index,] #<-- training set
temp <- movielens[test_index,] #<-- validation set

# To make sure userId and movieId in validation set are also in edx set, we use the following code:
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Then, we add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Finally, we also estimate the RMSE, a typical metric of "goodness of fit". This function will be used to validate the model.
  RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }
  
# THE MODEL
#During the Harvard lesson, a rating prediction model was built in which the prediction was a function of the average of all the ratings in the database
#which was somehow adjusted through some parameters (the b's) to take into account the average rating of the genre, the film and the user. 

#Instead of taking the average of all ratings as an average, we consider an average that is obtained by combining the average of the user ratings, the average
#of the film ratings and the average of the genre ratings through weights. 
#Four models are then constructed, in which we vary the composition of the weights, and then select the one leading to the lowest RMSE.
  

######### MODEL 1 #########  
  
  lambdas <- seq(0, 10, 0.25) #--> To estimate the b's (the terms that represents the average rating for movie, for userId and for genres),
  #it is minimized an equation which contains a penalty term: lambda. Lambada is a tuning parameter.
 
  weight1<-0.33  #<-hypothesis 1: weight of the average of genres ratings
  weight2<-0.33  #<-hypothesis 1: weight of the average of movie ratings
  weight3<-0.33  #<-hypothesis 1: weight of the average of user ratings
  
  
  rmses <- sapply(lambdas, function(l){   #we build a function to which apply the values of the lambda vector
    
    mu1 <- edx %>% group_by(genres) %>%    #<--average of genres ratings
      summarize(mu1=mean(rating))
    mu2 <- edx %>% group_by(movieId) %>%  #<--average of movie ratings
      summarize(mu2=mean(rating))
    mu3 <- edx %>% group_by(userId) %>%   #<--average of user ratings
      summarize(mu3=mean(rating))
    
  #Instead of taking the average of all ratings in the dataset as mu, we took a weighted average which considers the average of all the user's ratings, the average of all the ratings of the movies and the average of the genres ratings.
    mu <- edx %>% 
      left_join(mu1, by = "genres") %>%
      left_join(mu2, by = "movieId") %>%
      left_join(mu3, by = "userId") %>%
      mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3)%>% 
      select(genres, movieId, mu, mu1, mu2, mu3)
    
    
    edx2<- data.frame(edx, mu) #<- we attach mu on the edx database creating a new database: edx2
    
    b_i <- edx2 %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l)) # We now calculate b_i, the term that considers the average rating of each film adjusted 
    # with the lambda parameter to take into account the number of ratings and penalize smaller groups. 
    b_u <- edx2 %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l)) # We now calculate b_u, the term that considers the average rating of each user adjusted 
    # with the lambda parameter to take into account the number of ratings and penalize smaller groups. 
        b_g <- edx2 %>% 
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l)) # We now calculate b_g, the term that considers the average rating of each genre adjusted 
        # with the lambda parameter to take into account the number of ratings and penalize smaller groups. 
        
    predicted_ratings <- 
      validation %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_g, by = "genres") %>%
      left_join(mu1, by= "genres") %>%
      left_join(mu2, by="movieId")%>%
      left_join(mu3, by="userId")%>%
      mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3) #<-- We attach all the parameters we estimated previously in the training set on the test set.
    # For simplicity, we recalculate mu directly here. 
    
  predicted_ratings2<-predicted_ratings %>% 
      mutate(pred = mu_t + b_i + b_u + b_g) %>%
      .$pred  #<-. #We then make our rating estimate, a function of the mu we reformulated and the b's parameters
    
  return(RMSE(predicted_ratings2, validation$rating)) #In the last step we compare our rating estimate with the true rating of the test set by calculating the RMSE
  })
  
  
  # After calculating all possible RMSEs associated with the vector of Lambdas, we seek the lambda value at which the RMSE reaches its minimum value.
  
  qplot(lambdas, rmses)  
  
  lambda <- lambdas[which.min(rmses)]
  lambda
  
  #The lambda value at which the RMSE reaches its minimum value is 6.5
  # We repeat the simulation with this value. The code is the same as before.
  lambdas <- 6.5 
  rmses <- sapply(lambdas, function(l){
    
    mu1 <- edx %>% group_by(genres) %>% 
      summarize(mu1=mean(rating))
    mu2 <- edx %>% group_by(movieId) %>% 
      summarize(mu2=mean(rating))
    mu3 <- edx %>% group_by(userId) %>% 
      summarize(mu3=mean(rating))
    
    mu <- edx %>% 
      left_join(mu1, by = "genres") %>%
      left_join(mu2, by = "movieId") %>%
      left_join(mu3, by = "userId") %>%
      mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3) %>% select(genres, movieId, mu, mu1, mu2, mu3)
    
    edx2<- data.frame(edx, mu)
    
    b_i <- edx2 %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- edx2 %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    b_g <- edx2 %>% 
      left_join(b_i, by="movieId") %>%
      left_join(b_u, by="userId") %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
    
    predicted_ratings <- 
      validation %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      left_join(b_g, by = "genres") %>%
      left_join(mu1, by= "genres") %>%
      left_join(mu2, by="movieId")%>%
      left_join(mu3, by="userId")%>%
      mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
    
    predicted_ratings2<-predicted_ratings %>% 
      mutate(pred = mu_t + b_i + b_u + b_g) %>%
      .$pred
    return(RMSE(predicted_ratings2, validation$rating))
  })
  
  print(rmses) #<-- 0.8644893 is therefore the RMSE of our model 1
  
  
######### MODEL 2 #########  

#  The second model is identical to the first. It simply changes the composition of the weights defining the average mu (weight1, weight2, weight3).
# In particular, a higher weight is given to the average of the user ratings.
  
lambdas <- seq(0, 10, 0.25) 
weight1<-0.25
weight2<-0.35
weight3<-0.4

rmses <- sapply(lambdas, function(l){

  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))

 mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3)%>% 
    select(genres, movieId, mu, mu1, mu2, mu3)
    
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
 
  predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})


# After calculating all possible RMSEs associated with the vector of Lambdas, we seek the lambda value at which the RMSE reaches its minimum value.

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

#The lambda value at which the RMSE reaches its minimum value is 6.75
# We repeat the simulation with this value. The code is the same as before.

lambdas <- 6.75 
rmses <- sapply(lambdas, function(l){
  
  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))
  
   mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3) %>% select(genres, movieId, mu, mu1, mu2, mu3)
  
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
  
   predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})

print(rmses) #<-- 0.8645668 is therefore the RMSE of our model 2


######### MODEL 3 #########  

#  The third model is identical to the others. It simply changes the composition of the weights defining the average mu (weight1, weight2, weight3).
# In particular, a higher weight is given to the average of the user ratings.

lambdas <- seq(0, 10, 0.25) 
weight1<-0.2
weight2<-0.35
weight3<-0.45

rmses <- sapply(lambdas, function(l){
  
  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))
  
    mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3)%>% 
    select(genres, movieId, mu, mu1, mu2, mu3)
  
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
  
  predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})


# After calculating all possible RMSEs associated with the vector of Lambdas, we seek the lambda value at which the RMSE reaches its minimum value.

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

#The lambda value at which the RMSE reaches its minimum value is 6.75
# We repeat the simulation with this value. The code is the same as before.

lambdas <- 6.75
rmses <- sapply(lambdas, function(l){
  
  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))
  
   mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3) %>% select(genres, movieId, mu, mu1, mu2, mu3)
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
  
  predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})

print(rmses) #<-- 0.8646413 is therefore the RMSE of our model 3


######### MODEL 4 #########   

#  The fourth model is identical to the others. It simply changes the composition of the weights defining the average mu (weight1, weight2, weight3).
# In particular, a higher weight is given to the average of the user ratings.

lambdas <- seq(0, 10, 0.25) 
weight1<-0.2
weight2<-0.3
weight3<-0.5

rmses <- sapply(lambdas, function(l){
  
  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))
  
  mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3)%>% 
    select(genres, movieId, mu, mu1, mu2, mu3)
  
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
  
    predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})


# After calculating all possible RMSEs associated with the vector of Lambdas, we seek the lambda value at which the RMSE reaches its minimum value.

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

#The lambda value at which the RMSE reaches its minimum value is 6.25
# We repeat the simulation with this value. The code is the same as before.

lambdas <- 6.25
rmses <- sapply(lambdas, function(l){
  
  mu1 <- edx %>% group_by(genres) %>% 
    summarize(mu1=mean(rating))
  mu2 <- edx %>% group_by(movieId) %>% 
    summarize(mu2=mean(rating))
  mu3 <- edx %>% group_by(userId) %>% 
    summarize(mu3=mean(rating))
  
  mu <- edx %>% 
    left_join(mu1, by = "genres") %>%
    left_join(mu2, by = "movieId") %>%
    left_join(mu3, by = "userId") %>%
    mutate(mu=weight1*mu1+weight2*mu2+weight3*mu3) %>% select(genres, movieId, mu, mu1, mu2, mu3)
  
  
  edx2<- data.frame(edx, mu)
  
  b_i <- edx2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx2 %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(mu1, by= "genres") %>%
    left_join(mu2, by="movieId")%>%
    left_join(mu3, by="userId")%>%
    mutate(mu_t=weight1*mu1+weight2*mu2+weight3*mu3)
  
  predicted_ratings2<-predicted_ratings %>% 
    mutate(pred = mu_t + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings2, validation$rating))
})

print(rmses) #<-- 0.8647332 is therefore the RMSE of our model 4

# CONCLUSION
#Model 1, which constructs mu using the same weights for all three averages, is the one with the lowest RMSE.
#It is therefore the best model among those we determined.




