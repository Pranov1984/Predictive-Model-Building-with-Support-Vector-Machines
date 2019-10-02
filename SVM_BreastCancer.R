#Set working directory and import data
setwd("C:\\Users\\user\\Desktop\\Blogs\\SVM")
mydata=read.csv("Breast_Cancer_Dataset.csv")

#Check data structure and dimension
str(mydata)
dim(mydata)

#Change the levels of the target variable to "0" and "1" which stand for benign and malignant respectively
mydata$Class=ifelse(mydata$Class==2,0,1)
mydata$Class=as.factor(mydata$Class)
table(mydata$Class)

#Unsual level identified in Bare_Nuclei. Identify the rows and remove them
table(mydata$Bare_Nuclei)
which(mydata$Bare_Nuclei=="?")

data=mydata[-which(mydata$Bare_Nuclei=="?"),]
data=droplevels(data)
str(data)

#Partition the data in 70:30 ratio

library(caret)
set.seed(1234)
Index=createDataPartition(data$Class, p=0.7,list = FALSE)
Train=mydata[Index,]
Test=mydata[-Index,]


#Prepare for model by having 10 fold bootstrapped crossvalidation sampling
control=trainControl(method = "repeatedcv", number = 10, repeats = 1)

#Build a SVM model using Linear kernel
###Tuning parameter C for optimized model
grid=expand.grid(C = c(0.01, 0.02,0.05, 0.075, 0.1, 0.25, 0.5, 1, 1.25, 1.5, 1.75, 2,5))

set.seed(123456)
svm_Linear_Grid=train(Class~., data = Train,method = "svmLinear",
                      trControl = control, 
                      preProcess=c("scale","center"),
                      tuneGrid=grid)

a=svm_Linear_Grid$results
TrainingAcc_Linear=a[which.max(a$Accuracy),"Accuracy"]

#Best parameter value after tuning
svm_Linear_Grid$bestTune

#Predictions
Pred=predict(svm_Linear_Grid,Test)
a=confusionMatrix(Pred,Test$Class)
accuracy_Linear=a$overall[[1]]
accuracy_Linear
#An accuracy of 94.1% is achieved


#Visulize the confusion matrix
a$table
fourfoldplot(a$table)

Sensitivity_Linear=a$byClass[[1]]
#True positive rate achieved is 96%
Specificity_Linear=a$byClass[[2]]
#True negative rate achieved is 89%

###ROC and AUC
library(ROCR)
predictions.L=prediction(as.numeric(Pred),Test$Class)
Perf.L=performance(predictions.L,"tpr","fpr")
plot(Perf.L, main="ROC - SVM with Linear Kernel")
AUC=performance(predictions.L,"auc")
AUC_L=AUC@y.values
AUC_L


#Build the model by using Polynomial Kernel
#parameter tuning for both Cost and degree of polynomial
grid=expand.grid(C = c(0.005,.01, .1, 1,10), 
                 degree=c(2,3,4), scale=1)
svm_P=train(Class~.,data=Train,method="svmPoly",tuneLength=10,
            trControl=control, tuneGrid=grid)

#Best parameter value after tuning is a cost of 0.01 and degree of 2
svm_P$bestTune
#Visualize
plot(svm_P)


#Training Accuracy
b=svm_P$results
TrainingAcc_Poly=b[which.max(b$Accuracy),"Accuracy"]
TrainingAcc_Poly


#Predictions
Pred_Poly=predict(svm_P,Test)
b=confusionMatrix(Pred_Poly,Test$Class)
Accuracy_Polynomial=b$overall[[1]]
Accuracy_Polynomial
#An accuracy of 94.1% is achieved


#Visulize the confusion matrix
b$table
fourfoldplot(b$table)

Sensitivity_Polynomial=b$byClass[[1]]
#True positive rate achieved is 96%
Specificity_Polynomial=b$byClass[[2]]
#True negative rate achieved is 91%

##ROC& AUC
predictions.P=prediction(as.numeric(Pred_Poly),labels=Test$Class)
Perf.P=performance(predictions.P,"tpr","fpr")
plot(Perf.P, main="ROC - SVM with Polynomial Kernel")
AUC=performance(predictions.P,"auc")
AUC_P=AUC@y.values
AUC_P

#Build the model by using Radial Kernel
#parameter tuning for both Cost and sigma of radial kernel

grid=expand.grid(C = c(0.005,.01, 0.1, 0.15,0.20,0.25), 
                 sigma=c(0.0025,0.005,0.01,0.015,0.02,0.025))
set.seed(88888)
svm_Radial=train(Class~.,data=Train,method="svmRadial",tuneLength=10,
            trControl=control, tuneGrid=grid)

#Best tuned parameters for the model & Visualization of the comparison while tuning
svm_Radial$bestTune
plot(svm_Radial)

#Training Accuracy
c=svm_Radial$results
TrainingAcc_Rad=c[which.max(c$Accuracy),"Accuracy"]
TrainingAcc_Rad


#Predictions
Pred_Radial=predict(svm_Radial,Test)
c=confusionMatrix(Pred_Radial,Test$Class)
Accuracy_Radial=c$overall[[1]]
Accuracy_Radial
#An accurtacy of 96% is achieved

#Visulize the confusion matrix
c$table
fourfoldplot(c$table)

Sensitivity_Radial=c$byClass[[1]]
#True positive rate achieved is 96.5%
Specificity_Radial=c$byClass[[2]]
#True negative rate achieved is 95%

#ROC & AUC
predictions.R=prediction(as.numeric(Pred_Radial),labels=Test$Class)
Perf.R=performance(predictions.R,"tpr","fpr")
plot(Perf.R, main="ROC - SVM with Radial Kernel")
AUC=performance(predictions.R,"auc")
AUC_R=AUC@y.values
AUC_R

##Comparison across kernels
Compare=data.frame(Kernel=c("Linear","Poly","Radial"),
                   Train_Acc=c(TrainingAcc_Linear,TrainingAcc_Poly,TrainingAcc_Rad),
                   Test_Acc=c(accuracy_Linear,Accuracy_Polynomial,Accuracy_Radial),
                   Sensitivity=c(Sensitivity_Linear,Sensitivity_Polynomial,Sensitivity_Radial),
                   Specificity=c(Specificity_Linear,Specificity_Polynomial,Specificity_Radial),
                   AUC_All=c(AUC_L[[1]],AUC_P[[1]],AUC_R[[1]]))
Compare

