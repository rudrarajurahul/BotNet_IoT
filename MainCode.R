## Installing required libraries
library(ROCR)
library(reshape2)
library(e1071)
library(caret)
library(corrplot)
library(ggplot2)
library(reshape2)
library(dplyr)
require(MASS)
require(clusterGeneration)
library(logistf)
library(pscl)
library(caret)
library(arm)
library(Rfast)
library(standardize)
library(glmnet)
library(InformationValue)
library(mlbench)
library(dplyr)
library(caret)
library(rattle)                 # Fancy tree plot
library(rpart.plot)
library(dplyr)
library(parallel)
library(rpart)
library(randomForest)

#### Set working directory
setwd("/Users/sysadmin/Documents/Project")

## Loading dataset
data  = read.csv("Vinay_BoTNet_IoT.csv")

## Understanding the basic structure of dataset
head(data)
str(data)
summary(data)

## Checking for missing values
na <- apply(data,2,function(x) any(is.na(data)))
sum(na)

plot.new()
frame()

## Understanding the target variable
counts <- table(data$label)
print(counts)
#png("LabelDistribution.png")
barplot(counts,main='Label Distribution')

## Plotting the first column to understand distribution
png("DataDistribution.png")
plot(data$MI_dir_L5_weight,data$label)

## Lets seperate the data into 2 sets based on Labels - 0  and 1
data0 <- data[data$label == 0, ]
data1 <- data[data$label == 1, ]

## Plotting benign Vs Junk traffic distribution for each column
## Since there are a lot of columns, will run a loop and compare how both the label wise data is distributed
for (i in 1:ncol(data)) {
  png(file = paste("Comparetraffic - Column", i, ".jpg", sep=""))
  par(mfrow=c(1,2))
  plot(data0[, i],main=colnames(data0[i]),col='green')
  plot(data1[, i],main=colnames(data1[i]),col='red')
  dev.off()
}


## Lets look at the datatypes and the range of values in each column
str(data)

## Need also look at understanding each of the network protocols across the temporals 0.01, 0.1, 1, 3 and 5
## Dividing the dataset into protocol wise subsets to be able to plot
data_MI= data[ , grepl( "MI_dir" , names( data ) ) ]
data_H= data[ , grepl( "H" , names( data ) ) ]
data_HH= data[ , grepl( "HH" , names( data ) ) ]
data_HH_jit= data[ , grepl( "HH_jit" , names( data ) ) ]
data_HpHp= data[ , grepl( "HpHp" , names( data ) ) ]

## PCA to reduce dimensionality
## seperating target variable
data_pca <- data.frame(data)
str(data_pca)
#splitting into train and test for pca
require(caTools)
set.seed(101)
sample = sample.split(data_pca$ID, SplitRatio = .75)
train = subset(data_pca, sample == TRUE)
test = subset(data_pca, sample == FALSE)

## Removing target and identifier variables from train and test dataset
train_pca <- train
train_pca$ID <- NULL
train_pca$label <- NULL

## Saving test label into predictions dataframe 
predictions_df <- data.frame(test$label)

test_pca <- test
test_pca$label <- NULL
test_pca$ID <- NULL

## PCA for dimensionality reduction
prin_comp <- prcomp(train_pca, scale. = T)
names(prin_comp)

#mean of variables
prin_comp$center

#standard deviation of variables
prin_comp$scale

#principal component loading vectors
prin_comp$rotation
prin_comp$rotation[1:5,1:4]
dim(prin_comp$x)
png("PCA.png")
biplot(prin_comp, scale = 0)

## Checking the stddev and variance
std_dev <- prin_comp$sdev
pr_var <- std_dev^2

#check variance of first 10 components
pr_var[1:10]

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex[1:21]

#scree plot

plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b", main="Scree Plot")


#cumulative scree plot

plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b", main="Scree Plot Cumulative")

#Creating Training set with principal components
dim(prin_comp$x)
train.data <- data.frame(label = train$label, prin_comp$x)

dim(train.data)

#Selecting first 20 PCAs and adding target variable
train.data <- train.data[,1:21]
train.data$label <- train$label
train.data$label<-as.factor(train.data$label)
dim(train.data)
colnames(train.data)

#Convering test data into PCA
test.data <- predict(prin_comp, newdata = test_pca)
test.data <- as.data.frame(test.data)
colnames(test.data)
test.data <- test.data[,1:20]
test.data$label <- test$label
test.data$label<-as.factor(test.data$label)
dim(test.data)
colnames(test.data)

## Modelling with logistic regression
set.seed(1)
system.time(model_lg <- train(label ~., data=train.data, method='glm',
                              tuneGrid=expand.grid(parameter=c(1,10,100, 1000))))
model_lg
pred_train_lg <-predict(model_lg,train.data,type='raw')
pred_test_lg <- predict(model_lg,test.data,type='raw')
pred_test_lg <- as.numeric(pred_test_lg)
pred_test_lg[pred_test_lg=='1']<- 0
pred_test_lg[pred_test_lg=='2']<- 1
pred_test_lg

str(predictions_df)

#Confusion matrix
confusionMatrix(test$label,pred_test_lg)

# Accuracy
acc_lg<-1-misClassError(test$label,pred_test_lg)
results_df <- data.frame("Model"="Logistic","Accuracy"=acc_lg)
results_df

## Modelling with Decision tree
library(rpart)
system.time(model_dt <- rpart(label~., data = train.data))
model_dt
pred_train_dt <- predict(model_dt, train.data,type = 'prob')
pred_test_dt <- predict(model_dt, test.data,type='prob')
pred_test_dt <- data.frame(pred_test_dt)

str(pred_test_dt)

for (i in 1:nrow(pred_test_dt)){
  if(pred_test_dt[i,1] > pred_test_dt[i,2])
    {
  pred_test_dt[i,3] <- 0 
} 
  else
{
  pred_test_dt[i,3] <- 1
}
}
pred_test_dt$X0 <- NULL
pred_test_dt$X1 <- NULL

str(pred_test_dt)

predictions_df$DecisionTree <- pred_test_dt
predictions_df$Logistic <- pred_test_lg

confusionMatrix(test$label,pred_test_dt)

# Accuracy
acc_dt<-1-misClassError(test.data$label,pred_test_dt)
add <- c("Model"='DecisionTree',"Accuracy"=acc_dt)
results_df <- rbind(results_df,add)
results_df


## Modelling with RandomForest
library(NMF)
library(rpart)
library(randomForest)

model_rf <- randomForest(label~.,train.data, importance = TRUE)
model_rf
pred_train_rf <- predict(model_rf,train.data)
pred_test_rf <- predict(model_rf, test.data,type='response')

predictions_df
pred_test_rf <- as.numeric(pred_test_rf)
pred_test_rf[pred_test_rf=='1']<- 0
pred_test_rf[pred_test_rf=='2']<- 1
pred_test_rf
predictions_df$RandomForest <- pred_test_rf
confusionMatrix(test.data$label,pred_test_rf)

# Accuracy
acc_rf<-1-misClassError(test.data$label,pred_test_rf)
add <- c("Model"='Randomforest',"Accuracy"=acc_rf,)
results_df <- rbind(results_df,add)
results_df


## Ensembling

## Mojority voting

str(predictions_df)

predictions_df$ensembleVoting = as.factor(ifelse(predictions_df$Logistic==1 & predictions_df$DecisionTree==1,1,ifelse(predictions_df$RandomForest==1 & predictions_df$Logistic==1,1,ifelse(predictions_df$RandomForest==1 & predictions_df$DecisionTree==1,1,0))))
predictions_df
predictions_df$ensembleVoting <- as.numeric(predictions_df$ensembleVoting)
str(predictions_df)

predictions_df$ensembleVoting[predictions_df$ensembleVoting=='1']<- 0
predictions_df$ensembleVoting[predictions_df$ensembleVoting=='2']<- 1
predictions_df


confusionMatrix(predictions_df$test.label,predictions_df$ensembleVoting)

# Accuracy
acc_ev<-1-misClassError(test.data$label,predictions_df$ensembleVoting)
acc_ev
add <- c("Model"='Ensemble',"Accuracy"=acc_ev)
results_df <- rbind(results_df,add)
results_df
library(pROC)

## Plotting ROC/ AUC curves
roc1 = roc(test$label,pred_test_lg)
plot(roc1, main='ROC curve for Logistic Regression')

roc2 = roc(test$label,pred_test_dt$V3)
plot(roc2, main='ROC curve for Decision Tree')

roc3 = roc(test$label,pred_test_rf)
plot(roc3, main='ROC curve for Random Forest')

roc4 = roc(test$label,predictions_df$ensembleVoting)
plot(roc4, main='ROC curve for Ensemble')


results_df

