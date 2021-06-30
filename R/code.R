# Data Preparation
## Data reading
# Clear working environment
rm(list=ls())
# Set working directory
setwd("/Users/mayankanand/My Files/Edwisor - Data Science/Project1/")
# Check working directory
getwd()
# Install and Load Libraries
x = c("ggplot2","scales","corrplot","tidyverse","lubridate","caret","ROSE","pROC","ROCR","e1071")
lapply(x, install.packages, character.only = TRUE)
lapply(x, require, character.only = TRUE)
rm(x)
# Import Data from CSV
data = read.csv("bank-loan.csv", header = TRUE, na.strings = c(" ", "", "NA"))
# take a peek at the first 5 rows of the data
head(data)
#Viewing columns of data
colnames(data)
## EDA
# Explore data types and values in the data frame.
str(data)
# summarize attribute distributions
summary(data)
# dimensions of dataset
dim(data)
#Age - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=age),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=age)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=age,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Education - Bar Chart & Density Plot
ggplot(data) + geom_bar(aes(x=ed,color=default,fill=default), fill="white") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data, aes(x=ed)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white") + geom_density(,alpha=.2) 
#Employment Status - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=employ),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=employ)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=employ,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Address - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=address),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=address)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=address,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Income - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=income),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=income)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=income,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Debt Income - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=debtinc),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=debtinc)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=debtinc,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Credit to Debit Ratio - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=creddebt),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=creddebt)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=creddebt,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Other Debt Income - Histogram, Density Plot, Comparision Density Plot
ggplot(data) + geom_histogram(aes(x=othdebt),binwidth=5,fill="gray")
ggplot(data) + geom_density(aes(x=othdebt)) + scale_x_continuous(labels=number)
ggplot(data, aes(x=othdebt,color=default,fill=default)) + 
  geom_histogram(aes(y=..density..),binwidth=5, colour="black", fill="white") + geom_density(,alpha=.2) 
#Default - Dependent Variable - Bar Plots and Density Plot
ggplot(data) + geom_bar(aes(x=default,color=age,fill=age), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=ed,fill=ed), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=employ,fill=employ), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=address,fill=address), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=income,fill=income), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=debtinc,fill=debtinc), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=creddebt,fill=creddebt), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_bar(aes(x=default,color=othdebt,fill=othdebt), fill="gray") + theme(axis.text.y=element_text(size=rel(0.8)))
ggplot(data) + geom_histogram(aes(x=default), stat="count", binwidth=5, color="black", fill="gray")
# Missing Values Analysis
# Create dataframe with missing values of all columns
missing_val = data.frame(apply(data, 2, function(x){sum(is.na(x))}))
# Rename the column name
names(missing_val)[1] = "Missing_Percentage"
# Delete all missing value rows from the dataset and storing it in a new variable
row_nums = which(is.na(data$default))
data_nas = data[c(row_nums),]
# Create a data frame with all the missing values only
data = data[-c(row_nums),]
# Removing row indexes stored in a variable which is not required further
rm(row_nums)
# Outlier Analysis
# Box Plots - Distribution and Outlier Check
index = sapply(data,is.numeric)
print(index)
cnames = colnames(data)
# Plotting Box Plot for all the columns of the data
boxplot(data)
### Age - Doesn't have any outliers
boxplot(data$age)
### Education - The domain of the problem requires the values which are outliers for education box plot
boxplot(data$ed)
summary(data$ed)
Q1 = summary(data$ed)[2]
Q3 = summary(data$ed)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
boxplot(subset(data$ed,(data$ed<=5 & data$ed>Q1-1.5*IQR)))
#Checking the total number of rows present after removing each column's outliers
dim(data)
### Employment Status
boxplot(data$employ)
summary(data$employ)
Q1 = summary(data$employ)[2]
Q3 = summary(data$employ)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$employ,(data$employ<Q3+1.5*IQR & data$employ>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$employ<Q3+1.5*IQR & data$employ>Q1-1.5*IQR))
#Checking the total number of rows present after removing each column's outliers
dim(data)
#Address
boxplot(data$address)
summary(data$address)
Q1 = summary(data$address)[2]
Q3 = summary(data$address)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Changing the upper limit due to outliers still present in the data when upper limit is 3rd quartile
Q4 = quantile(data$address, 0.73)
IQR = Q4-Q1
print(c(Q4,IQR))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$address,(data$address<Q4+1.5*IQR & data$address>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$address<Q4+1.5*IQR & data$address>Q1-1.5*IQR))
# Getting the total no. of rows and columns in data
dim(data)
#Income
boxplot(data$income)
summary(data$income)
Q1 = summary(data$income)[2]
Q3 = summary(data$income)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Changing the upper limit due to outliers still present in the data when upper limit is 3rd quartile
Q4 = quantile(data$income, 0.68)
IQR = Q4-Q1
print(c(Q4,IQR))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$income,(data$income<Q4+1.5*IQR & data$income>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$income<Q4+1.5*IQR & data$income>Q1-1.5*IQR))
# Getting the total no. of rows and columns in data
dim(data)
#Debt Income
boxplot(data$debtinc)
summary(data$debtinc)
Q1 = summary(data$debtinc)[2]
Q3 = summary(data$debtinc)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Changing the upper limit due to outliers still present in the data when upper limit is 3rd quartile
Q4 = quantile(data$debtinc, 0.73)
IQR = Q4-Q1
print(c(Q4,IQR))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$debtinc,(data$debtinc<Q4+1.5*IQR & data$debtinc>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$debtinc<Q4+1.5*IQR & data$debtinc>Q1-1.5*IQR))
# Getting the total no. of rows and columns in data
dim(data)
#Credit to Debt Ratio
boxplot(data$creddebt)
summary(data$creddebt)
Q1 = summary(data$creddebt)[2]
Q3 = summary(data$creddebt)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Changing the upper limit due to outliers still present in the data when upper limit is 3rd quartile
Q4 = quantile(data$creddebt, 0.67)
IQR = Q4-Q1
print(c(Q4,IQR))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$creddebt,(data$creddebt<Q4+1.5*IQR & data$creddebt>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$creddebt<Q4+1.5*IQR & data$creddebt>Q1-1.5*IQR))
# Getting the total no. of rows and columns in data
dim(data)
#Other Debt Income
boxplot(data$othdebt)
summary(data$othdebt)
Q1 = summary(data$othdebt)[2]
Q3 = summary(data$othdebt)[5]
IQR = Q3-Q1
print(c(Q1,IQR,Q3))
#Changing the upper limit due to outliers still present in the data when upper limit is 3rd quartile
Q4 = quantile(data$othdebt, 0.69)
IQR = Q4-Q1
print(c(Q4,IQR))
#Plotting the Box Plot after removing outliers using IQR method
boxplot(subset(data$othdebt,(data$othdebt<Q4+1.5*IQR & data$othdebt>Q1-1.5*IQR)))
#Updating the removed rows in the data frame variable
data = subset(data,(data$othdebt<Q4+1.5*IQR & data$othdebt>Q1-1.5*IQR))
# Getting the total no. of rows and columns in data
dim(data)
# Feature Engineering
corr_matrix = cor(data)
round(corr_matrix, 2)
corrplot(corr_matrix, method="color", type = "full", tl.col = "black", p.mat = corr_matrix, insig = "p-value", tl.srt = 45)
ggplot(data, aes(x=income, y=employ)) + geom_point()
ggplot(data, aes(x=debtinc, y=othdebt)) + geom_point()
# Feature Scaling
cnames = colnames(data)
cnames
# Normalisation (val-min)/(max-min) - 0 to 1
normalize = function(x){
  return ((x - min(x, na.rm = TRUE))/(max(x, na.rm=TRUE) - min(x, na.rm=TRUE)))
}
norm_data = as.data.frame(apply(data[,c(1,3:8)],2,normalize))
data[,c(1,3:8)] = norm_data
head(data)
rm(norm_data)
summary(data)
# Handling Imbalanced Data
# Changing the categorical dependent feature to factor type
data$default = as.factor(data$default)
Defaulted = subset(data,(data$default==1))
Not_Defaulted = subset(data,(data$default==0))
dim(Defaulted)
dim(Not_Defaulted)
data = ROSE(default~., data=data,N=500,seed=51)$data
dim(data)
# Predictive Modelling and Evaluation
ind = sample(2,nrow(data),replace = TRUE, prob = c(0.7,0.3))
train = data[ind==1,]
test = data[ind==2,]
X_train = train[,c('age','ed','employ','address','income','debtinc','creddebt','othdebt')]
y_train = train[,'default']
X_test = test[,c('age','ed','employ','address','income','debtinc','creddebt','othdebt')]
y_test = test[,'default']
# Multiple Logistic Regression
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_glm = train(default~., data=train, method="glm", metric=metric, trControl=control)
model_glm$results
y_pred = predict(model_glm, X_test)
confusionMatrix(y_test, y_pred)
proba_pred = predict(model_glm, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Decision Tree
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_dt = train(default~., data=data, method="rpart", metric=metric, trControl=control)
model_dt$results
dt_pred = predict(model_dt, X_test)
confusionMatrix(y_test, dt_pred)
proba_pred = predict(model_dt, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# K Nearest Neighbours
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_knn = train(default~., data=data, method="knn", metric=metric, trControl=control)
model_knn$results
knn_pred = predict(model_knn, X_test)
confusionMatrix(y_test, knn_pred)
proba_pred = predict(model_knn, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Naive Bayes
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_nb = train(default~., data=data, method="naive_bayes", metric=metric, trControl=control)
model_nb$results
nb_pred = predict(model_nb, X_test)
confusionMatrix(y_test, nb_pred)
proba_pred = predict(model_nb, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Random Forest
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_rf = train(default~., data=data, method="rf", metric=metric, trControl=control)
model_rf$results
rf_pred = predict(model_rf, X_test)
confusionMatrix(y_test, rf_pred)
proba_pred = predict(model_rf, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Ada Boosting
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_ab = train(default~., data=data, method="adaboost", metric=metric, trControl=control)
model_ab$results
ab_pred = predict(model_ab, X_test)
confusionMatrix(y_test, ab_pred)
proba_pred = predict(model_ab, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Gradient Boosting
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_gb = train(default~., data=data, method="gbm", metric=metric, trControl=control)
model_gb$results
gb_pred = predict(model_gb, X_test)
confusionMatrix(y_test, gb_pred)
proba_pred = predict(model_gb, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Support Vector Machines
control = trainControl(method="repeatedcv", number=10,repeats=3)
metric = "Accuracy"
model_svm = train(default~., data=data, method="svmRadial", metric=metric, trControl=control)
model_svm$results
svm_pred = predict(model_svm, X_test)
confusionMatrix(y_test, svm_pred)
proba_pred = predict(model_svm, X_test, probability=TRUE, type="prob")
roc_score = roc(y_test, proba_pred[,2])
auc(roc_score)
prediction_roc = prediction(proba_pred[,2],y_test)
pred = performance(prediction_roc, "tpr", "fpr")
plot(pred)
abline(a=0,b=1)
auc = performance(prediction_roc, "auc")
auc = unlist(slot(auc, "y.values"))
auc = round(auc, 4)
legend(.6,.2,auc,title="AUC",cex=1.2)
# Summary and Conclusion
## From all the algorithms, Random Forest and Ada Boosting had the 100% Accuracy.
## So, they both are the models which can be used to predict most of the default cases trained on this dataset.