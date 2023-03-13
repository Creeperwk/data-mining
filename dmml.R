library(tidyverse)
library(ggplot2)
library(scales)
library(dplyr)
library(broom)
library(pROC)
library(readr)

#knn
group_19 <- read_csv("group_19.csv")
data <- na.omit(group_19)

data$class <- ifelse(data$y=='yes', 1, 0)
data <- data[,-c(2,3,4,5,8,9,10,13,15,21)]
data$default <- ifelse(data$default=='no',0,1)
data$housing <- ifelse(data$housing=='no',0,1)
data$loan <- ifelse(data$loan=='no',0,1)


set.seed(1)
n <- nrow(data)
ind1 <- sample(c(1:n),        floor(0.5*n)) 
ind2 <- sample(c(1:n)[-ind1], floor(0.25* n)) 
ind3 <- setdiff(c(1:n),c(ind1,ind2))

data.knn.train <- data[ind1,]
data.knn.valid <- data[ind2,]
data.knn.test <- data[ind3,]

library(class)
K <- c(1:15)
valid.knn.error <- c()
for (k in K){
  valid.knn.pred <- knn(data.knn.train, data.knn.valid, data.knn.train$class, k=k)
  valid.knn.error[k] <- mean(data.knn.valid$class != valid.knn.pred)
}

plot(K, valid.knn.error, type="b", ylab="validation error rate")

#k=5
k.opt <- which.min(valid.knn.error)

test.knn.pred <- knn(data.knn.train, data.knn.test, data.knn.train$class, k=k.opt)
table(data.knn.test$class,test.knn.pred)



#LDA
set.seed(2)
n <- nrow(data)
lda.ind <- sample(c(1:n), floor(0.7*n))
data.lda.train <- data[lda.ind,]
data.lda.test  <- data[-lda.ind,]

library(MASS)
data.lda <- lda(class~., data=data.lda.train)
data.lda 

data.lda.pred <- predict(data.lda)
ldahist(data = data.lda.pred$x[,1], g=data.lda.train$class)

dataset <- data.frame(Type=data.lda.train$class, lda=data.lda.pred$x)
ggplot(dataset, aes(x=LD1)) + 
  geom_density(aes(group=Type, colour=Type, fill=Type), alpha=0.3)

pred.LD1 <-rep("0",nrow(data.lda.train))
pred.LD1[data.lda.pred$x[,1] > 0.4] <- "1"
mean(pred.LD1!=data.lda.train$class)

#