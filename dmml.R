library(tidyverse)
library(ggplot2)
library(scales)
library(dplyr)
library(broom)
library(pROC)
library(readr)
library(plyr)
library(skimr)
library(knitr)
library(randomForest)
library(ROCR)
library(rpart)
library(rpart.plot)
library(e1071)

#knn
group_19 <- read_csv("group_19.csv")
data <- na.omit(group_19)

data$class <- ifelse(data$y=='yes', 1, 0)
data <- data[,-c(2,3,4,5,8,9,10,13,15,21)]
data$housing <- ifelse(data$housing=='no',0,1)
data$loan <- ifelse(data$loan=='no',0,1)


set.seed(1)
n <- nrow(data)
knn.ind1 <- sample(c(1:n),        floor(0.5*n)) 
knn.ind2 <- sample(c(1:n)[-knn.ind1], floor(0.25* n)) 
knn.ind3 <- setdiff(c(1:n),c(knn.ind1,knn.ind2))

data.knn.train <- data[knn.ind1,]
data.knn.valid <- data[knn.ind2,]
data.knn.test <- data[knn.ind3,]

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

#Bagging and random forest

trees.data<- group_19%>%
  dplyr::select(age,duration,loan,campaign,previous,emp.var.rate,y) # Select all the categorical variables and age
trees.data$y<- as.factor(trees.data$y)
trees.data$loan<- as.factor(trees.data$loan)

set.seed(3)

ggplot(trees.data, aes(loan, fill=y)) + geom_bar() +
  xlab("Have a personal loan or not") + ylab("Number of clients") +
  ggtitle("Number of clients per having a loan or not") +
  scale_fill_discrete(name = "", labels = c("Faliure", "Success")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

n <- nrow(trees.data)
idx <- sample(1:n, round(0.2*n))
rf.valid <- trees.data[idx,]
rf.train <- trees.data[-idx,]
rf.train.imputed <- na.roughfix(rf.train) 
bagging <- randomForest(y~age + duration +  loan + campaign  + previous + emp.var.rate, data=rf.train.imputed,
                        mtry=4, ntree=200)
rf <- randomForest(y~age + duration +  campaign  +  loan +previous + emp.var.rate, data=rf.train.imputed,
                   ntree=200)

rf.valid.imputed <- na.roughfix(rf.valid)
bagging_prob <- predict(bagging, rf.valid.imputed, type="prob")
rf_prob <- predict(rf, rf.valid.imputed, type="prob")

bagging_pred <- prediction(bagging_prob[,2], rf.valid$y)
bagging_AUC  <- performance(bagging_pred, "auc")@y.values[[1]]
rf_pred <- prediction(rf_prob[,2], rf.valid$y)
rf_AUC  <- performance(rf_pred, "auc")@y.values[[1]]
print(c(bagging_AUC,rf_AUC))

varImpPlot(rf, main="Predicting success in different variables")

#Trees
set.seed(4)
n <- nrow(trees.data)
trees.ind1 <- sample(c(1:n), round(n/2))
trees.ind2 <- sample(c(1:n)[-trees.ind1], round(n/4))
trees.ind3 <- setdiff(c(1:n),c(trees.ind1,trees.ind2))
trees.train <- trees.data[trees.ind1, ]
trees.valid <- trees.data[trees.ind2, ]
trees.test  <- trees.data[trees.ind2, ]

trees.data.rt <- rpart(y~age + duration +   loan + campaign  + previous + emp.var.rate, data=trees.train, method="class")
rpart.plot(trees.data.rt,type=2,extra=4)

# training performance
train.pred <- predict(trees.data.rt, newdata=trees.train[,-7],type="class")
table(trees.train$y, train.pred)

# validation performance
valid.pred <- predict(trees.data.rt, newdata=trees.valid[,-7],type="class")
table(trees.valid$y, valid.pred)

train.table <- table(trees.train$y, train.pred)
train.table[1,1]/sum(train.table[1,]) # training sensitivity
train.table[2,2]/sum(train.table[2,]) # training specificity
valid.table <- table(trees.valid$y, valid.pred)
valid.table[1,1]/sum(valid.table[1,]) # validation sensitivity
valid.table[2,2]/sum(valid.table[2,]) # validation specificity

#Full tree
Full_tree <- rpart(y~age + duration +   loan + campaign  + previous + emp.var.rate, data=trees.train, method="class",
                   control=rpart.control(minsplit=2,minbucket=1,maxdepth=30,cp=-1))
printcp(Full_tree)

# prune the tree
trees.rt.pruned <- prune(Full_tree, cp=0.012)
rpart.plot(trees.rt.pruned)

# training performance
train.pred <- predict(trees.data.rt, newdata=trees.train[,-7],type="class")
train.table <- table(trees.train$y, train.pred)
train.table

# validation performance
valid.pred <- predict(trees.data.rt, newdata=trees.valid[,-7],type="class")
valid.table <- table(trees.valid$y, valid.pred)
valid.table


#Supporting vector machines and Kernelisation
set.seed(5)

svm.ind1 <- sample(c(1:n),        floor(0.5*n)) 
svm.ind2 <- sample(c(1:n)[-svm.ind1], floor(0.25* n)) 
svm.ind3 <- setdiff(c(1:n),c(svm.ind1,svm.ind2))
data.svm.train <- data[svm.ind1,]
data.svm.valid <- data[svm.ind2,]
data.svm.test <- data[svm.ind3,]

pred.error<-function(pred,truth){
  mean(pred!=truth)
}
C.val <- c(0.1,0.5,1,2,5,10)
C.error <- numeric(length(C.val))

for (i in 1:length(C.val)) {
  model <- svm(class~age+loan+duration+emp.var.rate,data=data.svm.train,type="C-classification",kernel="linear",cost=C.val[i]) #kernel will be explained in the next section
  pred.model <- predict(model, data.svm.valid)
  C.error[i] <- pred.error(pred.model, data.svm.valid$sp)
}
C.sel <- C.val[min(which.min(C.error))]
C.sel