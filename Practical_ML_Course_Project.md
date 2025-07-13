---
title: "Activity Classification using Wearable Sensors"
author: "Lee Jung Min"
date: "13. July 2025"
output:
  html_document:
    keep_md: yes
---

##Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: 
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har
 (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


## data, libraries

Loading data and library


``` r
library(lattice)
library(ggplot2)
library(caret)
library(kernlab)
```

```
## 
## Attaching package: 'kernlab'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     alpha
```

``` r
library(rattle)
```

```
## Loading required package: tibble
```

```
## Loading required package: bitops
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

``` r
library(corrplot)
```

```
## corrplot 0.95 loaded
```

``` r
library(randomForest)
```

```
## randomForest 4.7-1.2
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

``` r
set.seed(1234)

download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")

traincsv <- read.csv("pml-training.csv")
testcsv <- read.csv("pml-testing.csv")

dim(traincsv)
## [1] 19622   160
dim(testcsv)
## [1]  20 160
```

The training dataset contains 160 variables and 19,622 observations, while the test dataset also includes 160 variables but only 20 observations.

## Data processing(cleaning)
Remove Unnecessary variables from N/A variables.
1~7 colomn removing


``` r
traincsv <- traincsv[,colMeans(is.na(traincsv)) < .95] #removing na
traincsv <- traincsv[,-c(1:7)] #removing metadata
```

Removing near zero variance variables.


``` r
nvz <- nearZeroVar(traincsv)
traincsv <- traincsv[,-nvz]
dim(traincsv)
```

```
## [1] 19622    53
```

## Cross-validation
 training (80%) and testing (20%) data.


``` r
Samples <- createDataPartition(y=traincsv$classe, p=0.80, list=FALSE)
Training <- traincsv[Samples, ] 
Testing <- traincsv[-Samples, ]
```

## Prediction models
Decision Trees, Random Forest, Gradient Boosted Trees, and SVM.
Set up control to use 3-fold cross validation.


``` r
control <- trainControl(method="cv", number=3, verboseIter=F)
```

### Decision Tree


``` r
mod_trees <- train(classe~., data=Training, method="rpart", trControl = control, tuneLength = 5)
fancyRpartPlot(mod_trees$finalModel)
```

![](Practical_ML_Course_Project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

Prediction:


```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1023  313  309  301  103
##          B   16  246   25    7   77
##          C   58   83  301   85  101
##          D   19  117   49  250  124
##          E    0    0    0    0  316
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5445          
##                  95% CI : (0.5287, 0.5602)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4061          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9167  0.32411  0.44006  0.38880  0.43828
## Specificity            0.6345  0.96049  0.89904  0.90579  1.00000
## Pos Pred Value         0.4993  0.66307  0.47930  0.44723  1.00000
## Neg Pred Value         0.9504  0.85557  0.88376  0.88317  0.88772
## Prevalence             0.2845  0.19347  0.17436  0.16391  0.18379
## Detection Rate         0.2608  0.06271  0.07673  0.06373  0.08055
## Detection Prevalence   0.5223  0.09457  0.16008  0.14249  0.08055
## Balanced Accuracy      0.7756  0.64230  0.66955  0.64730  0.71914
```

### Random forest

``` r
mod_rf <- train(classe~., data=Training, method="rf", trControl = control, tuneLength = 5)

pred_rf <- predict(mod_rf, Testing)
cmrf <- confusionMatrix(pred_rf, factor(Testing$classe))
cmrf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    5    0    0    0
##          B    0  753    4    0    0
##          C    0    1  680   10    2
##          D    0    0    0  632    3
##          E    0    0    0    1  716
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9934          
##                  95% CI : (0.9903, 0.9957)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9916          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9942   0.9829   0.9931
## Specificity            0.9982   0.9987   0.9960   0.9991   0.9997
## Pos Pred Value         0.9955   0.9947   0.9812   0.9953   0.9986
## Neg Pred Value         1.0000   0.9981   0.9988   0.9967   0.9984
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1919   0.1733   0.1611   0.1825
## Detection Prevalence   0.2858   0.1930   0.1767   0.1619   0.1828
## Balanced Accuracy      0.9991   0.9954   0.9951   0.9910   0.9964
```

## Conclusion

### Result
```
##      accuracy oos_error
## Tree    0.5445    0.4555
## Random  0.9941    0.0059
```
### Expected out-of-sample error

The confusion matrices show, that the Random Forest algorithm performens better than decision trees. The accuracy for the Random Forest model was 0.9941 (95% CI: (0.9912, 0.9963)) compared to 0.5445 (95% CI: (0.5287, 0.5602)) for Decision Tree model. The random Forest model is better.

## Test set prediction

``` r
pred <- predict(mod_rf, testcsv)
print(pred)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
## plot

``` r
plot(mod_trees)
```

![](Practical_ML_Course_Project_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

Initially, the decision tree model showed an accuracy of about 54.45%, which improved significantly to 99.41% with the random forest model. Based on this increase, we expect the out-of-sample error to be very low—approximately 0.6%—indicating that the random forest model generalizes well to unseen data with minimal misclassification.
