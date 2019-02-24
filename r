

#Set working directory and import data
setwd("working directory")
mydata=read.csv("Breast_Cancer_Dataset.csv")

#Check data structure and dimension
str(mydata)
## 'data.frame':    699 obs. of  10 variables:
##  $ Clump_Thick                : int  5 5 3 6 4 8 1 2 2 4 ...
##  $ Uniformity_of_Cell_Size    : int  1 4 1 8 1 10 1 1 1 2 ...
##  $ Uniformity_of_Cell_Shape   : int  1 4 1 8 1 10 1 2 1 1 ...
##  $ Marginal_Adhesion          : int  1 5 1 1 3 8 1 1 1 1 ...
##  $ Single_Epithelial_Cell_Size: int  2 7 2 3 2 7 2 2 2 2 ...
##  $ Bare_Nuclei                : Factor w/ 11 levels "?","1","10","2",..: 2 3 4 6 2 3 3 2 2 2 ...
##  $ Bland_Chromatin            : int  3 3 3 3 3 9 3 3 1 2 ...
##  $ Normal_Nucleoli            : int  1 2 1 7 1 7 1 1 1 1 ...
##  $ Mitoses                    : int  1 1 1 1 1 1 1 1 5 1 ...
##  $ Class                      : int  2 2 2 2 2 4 2 2 2 2 ...
dim(mydata)
## [1] 699  10
#Change the levels of the target variable to "0" and "1" which stand for benign and malignant respectively
mydata$Class=ifelse(mydata$Class==2,0,1)
mydata$Class=as.factor(mydata$Class)
table(mydata$Class)
## 
##   0   1 
## 458 241
#Unsual level identified in Bare_Nuclei. Identify the rows and remove them
table(mydata$Bare_Nuclei)
## 
##   ?   1  10   2   3   4   5   6   7   8   9 
##  16 402 132  30  28  19  30   4   8  21   9
which(mydata$Bare_Nuclei=="?")
##  [1]  24  41 140 146 159 165 236 250 276 293 295 298 316 322 412 618
data=mydata[-which(mydata$Bare_Nuclei=="?"),]
data=droplevels(data)
str(data)
## 'data.frame':    683 obs. of  10 variables:
##  $ Clump_Thick                : int  5 5 3 6 4 8 1 2 2 4 ...
##  $ Uniformity_of_Cell_Size    : int  1 4 1 8 1 10 1 1 1 2 ...
##  $ Uniformity_of_Cell_Shape   : int  1 4 1 8 1 10 1 2 1 1 ...
##  $ Marginal_Adhesion          : int  1 5 1 1 3 8 1 1 1 1 ...
##  $ Single_Epithelial_Cell_Size: int  2 7 2 3 2 7 2 2 2 2 ...
##  $ Bare_Nuclei                : Factor w/ 10 levels "1","10","2","3",..: 1 2 3 5 1 2 2 1 1 1 ...
##  $ Bland_Chromatin            : int  3 3 3 3 3 9 3 3 1 2 ...
##  $ Normal_Nucleoli            : int  1 2 1 7 1 7 1 1 1 1 ...
##  $ Mitoses                    : int  1 1 1 1 1 1 1 1 5 1 ...
##  $ Class                      : Factor w/ 2 levels "0","1": 1 1 1 1 1 2 1 1 1 1 ...
#Partition the data in 70:30 ratio

library(caret)
## Warning: package 'caret' was built under R version 3.4.4
## Loading required package: lattice
## Loading required package: ggplot2
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
##     C
## 5 0.1
#Predictions
Pred=predict(svm_Linear_Grid,Test)
a=confusionMatrix(Pred,Test$Class)
accuracy_Linear=a$overall[[1]]
accuracy_Linear
## [1] 0.9409091
#An accuracy of 94.1% is achieved


#Visulize the confusion matrix
a$table
##           Reference
## Prediction   0   1
##          0 141   8
##          1   5  66
fourfoldplot(a$table)

accuracy_Linear=a$overall[[1]]

Sensitivity_Linear=a$byClass[[1]]
#True positive rate achieved is 96%
Specificity_Linear=a$byClass[[2]]
#True negative rate achieved is 89%

###ROC and AUC
library(ROCR)
## Loading required package: gplots
## 
## Attaching package: 'gplots'
## The following object is masked from 'package:stats':
## 
##     lowess
predictions.L=prediction(as.numeric(Pred),Test$Class)
Perf.L=performance(predictions.L,"tpr","fpr")
plot(Perf.L, main="ROC - SVM with Linear Kernel")

AUC=performance(predictions.L,"auc")
AUC_L=AUC@y.values
AUC_L
## [[1]]
## [1] 0.9288227
#Build the model by using Polynomial Kernel
#parameter tuning for both Cost and degree of polynomial
grid=expand.grid(C = c(0.005,.01, .1, 1,10), 
                 degree=c(2,3,4), scale=1)
svm_P=train(Class~.,data=Train,method="svmPoly",tuneLength=10,
            trControl=control, tuneGrid=grid)

#Best parameter value after tuning is a cost of 0.01 and degree of 2
svm_P$bestTune
##   degree scale    C
## 4      2     1 0.01
#Visualize
plot(svm_P)

#Training Accuracy
b=svm_P$results
TrainingAcc_Poly=b[which.max(a$Accuracy),"Accuracy"]
TrainingAcc_Poly
## numeric(0)
#Predictions
Pred_Poly=predict(svm_P,Test)
b=confusionMatrix(Pred_Poly,Test$Class)
Accuracy_Polynomial=b$overall[[1]]
Accuracy_Polynomial
## [1] 0.9409091
#An accuracy of 94.1% is achieved


#Visulize the confusion matrix
b$table
##           Reference
## Prediction   0   1
##          0 140   7
##          1   6  67
fourfoldplot(b$table)

accuracy_Linear=a$overall[[1]]

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
## [[1]]
## [1] 0.9321548
#Build the model by using Radial Kernel
#parameter tuning for both Cost and sigma of radial kernel

grid=expand.grid(C = c(0.005,.01, 0.1, 0.15,0.20,0.25), 
                 sigma=c(0.0025,0.005,0.01,0.015,0.02,0.025))
set.seed(88888)
svm_Radial=train(Class~.,data=Train,method="svmRadial",tuneLength=10,
            trControl=control, tuneGrid=grid)

#Best tuned parameters for the model & Visualization of the comparison while tuning
svm_Radial$bestTune
##    sigma   C
## 15  0.01 0.1
plot(svm_Radial)

#Training Accuracy
c=svm_Radial$results
TrainingAcc_Rad=c[which.max(a$Accuracy),"Accuracy"]
TrainingAcc_Rad
## numeric(0)
#Predictions
Pred_Radial=predict(svm_Radial,Test)
c=confusionMatrix(Pred_Radial,Test$Class)
Accuracy_Radial=c$overall[[1]]
Accuracy_Radial
## [1] 0.9590909
#An accurtacy of 96% is achieved

#Visulize the confusion matrix
c$table
##           Reference
## Prediction   0   1
##          0 141   4
##          1   5  70
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
## [[1]]
## [1] 0.9558497
##Comparison across kernels
Compare=data.frame(Kernel=c("Linear","Poly","Radial"),
                   Train_Acc=c(TrainingAcc_Linear,TrainingAcc_Poly,TrainingAcc_Rad),
                   Test_Acc=c(accuracy_Linear,Accuracy_Polynomial,Accuracy_Radial),
                   Sensitivity=c(Sensitivity_Linear,Sensitivity_Polynomial,Sensitivity_Radial),
                   Specificity=c(Specificity_Linear,Specificity_Polynomial,Specificity_Radial),
                   AUC_All=c(AUC_L[[1]],AUC_P[[1]],AUC_R[[1]]))
Compare
##   Kernel Train_Acc  Test_Acc Sensitivity Specificity   AUC_All
## 1 Linear 0.9792481 0.9409091   0.9657534   0.8918919 0.9288227
## 2   Poly 0.9792481 0.9409091   0.9589041   0.9054054 0.9321548
## 3 Radial 0.9792481 0.9590909   0.9657534   0.9459459 0.9558497
