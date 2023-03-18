library(tidyverse)
library(ggplot2)
library(scales)
library(dplyr)
library(MASS)
library(class)
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
library(neuralnet)
library(NeuralNetTools)
library(tidytext)

#dataset
group_19 <- read_csv("group_19.csv")

#information visualization
ggplot(group_19, aes(marital, fill=y)) + geom_bar() +
  xlab("Marital") + ylab("Number of clients") +
  ggtitle("Number of clients with different marital") +
  scale_fill_discrete(name = "", labels = c("Faliure", "Success")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

ggplot(group_19, aes(default, fill=y)) + geom_bar() +
  xlab("Credit in default") + ylab("Number of clients") +
  ggtitle("Number of clients with credit in default") +
  scale_fill_discrete(name = "", labels = c("Faliure", "Success")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

ggplot(group_19, aes(housing, fill=y)) + geom_bar() +
  xlab("Housing loan") + ylab("Number of clients") +
  ggtitle("Number of clients with a housing loan or not") +
  scale_fill_discrete(name = "", labels = c("Faliure", "Success")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

ggplot(group_19, aes(loan, fill=y)) + geom_bar() +
  xlab("Have a personal loan or not") + ylab("Number of clients") +
  ggtitle("Number of clients with a personal loan or not") +
  scale_fill_discrete(name = "", labels = c("Faliure", "Success")) +
  theme(plot.title = element_text(hjust = 0.5, size = 10))

#data cleaning
min.max.scale<- function(x){
  (x-min(x))/(max(x)-min(x))
  }

data <- na.omit(group_19) %>%
  dplyr::select(y,age,marital,loan,duration,cons.price.idx) %>%
  mutate(loan=as.factor(loan),y=as.factor(y)) %>%
  as.data.frame()

data <- cbind(data,model.matrix(~marital-1, data=data))

data <- data[,-3] %>%
  mutate_if(.predicate=is.numeric,
            .funs=min.max.scale)%>%
  as.data.frame()
  
data$loan<-ifelse(data$loan=="no",0,1)

#split into train set and text set
set.seed(123)
n <- nrow(data)
ind1 <- sample(c(1:n),        floor(0.5*n)) 
ind2 <- sample(c(1:n)[-ind1], floor(0.25* n)) 
ind3 <- setdiff(c(1:n),c(ind1,ind2))

data.train <- data[ind1,]
data.valid <- data[ind2,]
data.test <- data[ind3,]

##knn
library(kknn)

knn<- kknn(y~.,train = data.train,test = data.test)
summary(knn)

class.rate<-numeric(25)
for(k in 1:25) {
  pred.class <- knn(data.train[,-1], data.valid[,-1], data.train[,1], k=k)
  class.rate[k] <- sum(pred.class==data.valid[,1])/length(pred.class)
}
plot(c(1:25), class.rate, type="b",
     main="Correct classification rates on the validation data for a range of k",
     xlab="k",ylab="Correct Classification Rate",cex.main=0.7)

k.opt <- which.max(class.rate)

knn.pred <- knn(data.train[,-1], data.test[,-1], data.train[,1], k=k.opt)
table(data.test[,1],knn.pred)

##LDA
lda <- lda(y~age+loan+duration+cons.price.idx, data=data.train)

lda.pred <- predict(lda,newdata= data.test)$class
table(data.test$y,lda.pred)

##QDA
qda <- qda(y~age+loan+duration+cons.price.idx, data=data.train)

qda.pred<- predict(qda,newdata= data.test)$class
table(data.test$y,qda.pred)

##Bagging and random forest
bagging<- randomForest(y~.,data = data.train,mtry=4,ntree=200)
rf <- randomForest(y~., data=data.train,ntree=200)

bagging.pred <- predict(bagging, data.test, type="class")
rf.pred <- predict(rf, data.test, type="class")

table(data.test$y,bagging.pred)
table(data.test$y,rf.pred)

##Trees
tree <- rpart(y~., data=data.train, method="class")
rpart.plot(tree,type=2,extra=4)

tree.pred <- predict(tree, newdata=data.test[,-1],type="class")
table(data.test$y, tree.pred)


#Full tree
set.seed(1)
full.tree <- rpart(y~., data=data.train, method="class",
                   control=rpart.control(minsplit=2,minbucket=1,maxdepth=30,cp=-1))
printcp(full.tree)
plotcp(full.tree)

tree.pruned <- prune(full.tree, cp=0.011)
rpart.plot(tree.pruned)

tree.pruned.pred <- predict(tree.pruned, newdata=data.test[,-1],type="class")
table(data.test$y, tree.pruned.pred)


##Supporting vector machines and Kernelisation
#svm<-svm(y~age+duration,data=data.train,kernel='linear',type="C-classification")
#svm.pred<-predict(svm,newdata=data.test,type='class')

#table(data.test$y,svm.pred)

##Neural Networks
nn1<-neuralnet(y~.,data=data.train,hidden=3,linear.output=F,err.fct = 'ce',
               likelihood=TRUE, threshold = 0.1)
nn2<-neuralnet(y~.,data=data.train,hidden=5,linear.output=F,err.fct = 'ce',
               likelihood=TRUE, threshold = 0.1)
nn3<-neuralnet(y~.,data=data.train,hidden=7,linear.output=F,err.fct = 'ce',
               likelihood=TRUE, threshold = 0.1)

nn.class <- tibble('Network' = rep(c("NN_3","NN_5", "NN_7"), each = 3),
                       'Metric' = rep(c('AIC', 'BIC','CE loss'), length.out=9),
                       'Value' = c(nn1$result.matrix[4,1],
                                   nn1$result.matrix[5,1],
                                   nn1$result.matrix[1,1],
                                   nn2$result.matrix[4,1],
                                   nn2$result.matrix[5,1],
                                   nn2$result.matrix[1,1],
                                   nn3$result.matrix[4,1],
                                   nn3$result.matrix[5,1],
                                   nn3$result.matrix[1,1]))
nn_ggplot <- nn.class %>%
  ggplot(aes(Network, Value, fill=Metric)) +
  geom_col(position = 'dodge')  +
  ggtitle("AIC, BIC, and cross entropy loss of the neural networks")
nn_ggplot

nn<-nn2
plot(nn)

nn.prob<-predict(nn,newdata=data.test)
nn.pred<-ifelse(nn.prob[,2]>0.5,'yes','no')
table(data.test$y,nn.pred)


#Accuracy, precision, recall, F1-score
binary.class.metric <- function(true,predict,positive_level){
  accuracy = mean(true==predict)
  precision = sum(true==positive_level & predict==positive_level)/sum(predict==positive_level)
  recall = sum(true==positive_level & predict==positive_level)/sum(true==positive_level)
  fl_score = 2*precision*recall/(precision+recall)
  return(list(accuracy = accuracy,
              precision = precision,
              recall = recall,
              fl_score = fl_score))
}

knn.metric<-binary.class.metric(true=data.test$y,predict=knn.pred,positive_level='yes')
knn.metric

lda.metric<-binary.class.metric(true=data.test$y,predict=lda.pred,positive_level='yes')
lda.metric

qda.metric<-binary.class.metric(true=data.test$y,predict=qda.pred,positive_level='yes')
qda.metric

bagging.metric<-binary.class.metric(true=data.test$y,predict=bagging.pred,positive_level='yes')
bagging.metric

rf.metric<-binary.class.metric(true=data.test$y,predict=rf.pred,positive_level='yes')
rf.metric

tree.metric<-binary.class.metric(true=data.test$y,predict=tree.pred,positive_level='yes')
tree.metric


nn.metric<-binary.class.metric(true=data.test$y,predict=nn.pred,positive_level='yes')
nn.metric

#visualization
bind_rows(unlist(knn.metric),
          unlist(lda.metric),
          unlist(qda.metric),
          unlist(bagging.metric),
          unlist(rf.metric),
          unlist(tree.metric),
          unlist(nn.metric))%>%
  mutate(model=c('KNN','LDA','QDA','Bagging','Random Forest','Decision Tree','Neural Network'))%>%
  pivot_longer(cols=-model,
               names_to = 'metric',
               values_to = 'value')%>%
  mutate(model = reorder_within(x = model,by = value,within = metric)) %>%
  ggplot(aes(x = model,y = value,fill = metric)) +
  geom_col() +
  scale_x_reordered() +
  facet_wrap(~metric,scales = 'free') +
  labs(x ='Model',
       y ='Value',
       fill = 'Model') +
  coord_flip() +
  theme_test() 
