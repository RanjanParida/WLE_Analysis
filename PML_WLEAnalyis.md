# PML_WLEAnalysis
Ranjan Parida  
Friday, September 21, 2014  

# INTRODUCTION
## Context
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to build a model to predict "how well" the excercise was performed. 

## Data Collection Process
**Setup**  
Six participant, fitted with "on-body" sensors and also observed through "ambient sensing", we asked to perform barbell lifts correctly and incorrectly in 5 different ways. The "on-body" sensors were attached to the dumbbell, lumbar belt, glove and arm band of the participants. The "ambient sensor" used was microsoft kinect.

**Participant Demographics**  
All participants in the experiment were in the age range of 20 to 28 years with little weight lifting experience.

**Experiment**  
Participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions:  
* Class A - exactly according to the specifcation  
* Class B - throwing the elbows to the front  
* Class C - lifting the dumbbell only halfway  
* Class D - lowering the dumbbell only halfway  
* Class E - throwing the hips to the front  

These excercises was performed under the supervision of experienced weight lifters.

**Actual Data Collection**  
Features were collecting using a sliding window approach with different lengths from 0.5 second to 2.5 seconds, with 0.5 second overlap. In each step of the sliding window approach features on the Euler angles (roll, pitch and yaw) were calculated. Also, raw accelerometer, gyroscope and magnetometer readings were observed. For the Euler angles of each of the four sensors, eight features were calculated: mean, variance, standard deviation, max, min, amplitude, kurtosis and skewness, generating in total 96 derived feature sets.

More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

## Scope
Predicting the "how well" the weight lifting excercise (specifically, lifting of dumbbell) was performed based on the "accelerometer" data.

#ANALYSIS
## Training Data
The data from the above experiment has been made available to us in a [training dataset][1] from Course website. This data is downloaded and loaded to R for analysis.


```r
setInternet2(TRUE)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv" )
setInternet2(FALSE)
```

```
## [1] TRUE
```

```r
training <- read.csv("pml-training.csv")
```
## Test Set for Cross Validation
A subset of obeservations from the training data set is kept aside for the purposes of validation of the final model, to refine and adjust if needed. This subset is created by using a random sampling.

```r
library(caret)
set.seed(91488)
inTrain <- createDataPartition(y = training$classe, p=0.75, list=FALSE)
trainData <- training[inTrain,]
vTestData <- training[-inTrain,]
```

## Feature Selection
As the measured and some of the calculated data has already been made available to us in the training data set, we start the process of feature selection by performing an analysis on the summary of predictors.

```r
summary(trainData)
```

A quick look at the summary (Appendix A) shows their are 160 variables in the training data set, many of the which have over 95% of the values as NAs. 

### Removing variable with high number of observations with NA
There is no use of using variable with high number of observations with a value of **NA**. So, we eliminate variables with over 95% values as NA from further analysis.

This is accomplished by writing a simple function **rmvNACols** that takes the data set, threshold for elimination as percentage or number as input, and returns a dataset after eliminating the columns that exceeded the threshold. Details of this function has been listed in **Appendix B**. 



```r
trainData2 <- rmvNAcols(trainData, pct=.95)
```

### Removing variables with near zero variance
After removing the variables with high volume of NAs, we are still left with 93 columns. These are still a very high number of variables. These may all be relevant, or may be not. So, we look for variables that have near zero variance and thus would not be contributing a lot of value to the prediction model.


```r
nsv <- nearZeroVar(trainData2, saveMetrics=TRUE)
trainData3 <- trainData2[,rownames(nsv[nsv$nzv==FALSE,])]
```

Even after removing near zero variance predictors, we are still left with 58 

### Excluding non-accelerometer data from the variables

```r
colnames(trainData3[,1:10])
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"
```
As you would note, the first six columns in the remaining variables of the data set are either row identifier, or individual identifier, or timestamp/recording window. Since our task is to decipher if "how well" an excercise is performed based on accelerometer data, we will drop these first six variables as well.


```r
trainData4 <- trainData3[,-c(1:6)]
```

We are now left with a data set of 53 columns - 52 predictors and 1 outcome. As our **outcome is categorical** in nature, use of alogrithms like **decision tree or random forrest** make the most sense.

## Algorithm and Parameter Selection
Through the lecture notes, it is evident that random forrest has a better accuracy. Knowing we need to use this model to predict responses on the test data set for grading, it is evident that we are looking at the **accuracy as a measure** for model selection.


```r
library(randomForest)
set.seed(999)
modelFit1 <- randomForest(classe ~ ., data = trainData4, ntree = 1)
aCM <- confusionMatrix(predict(modelFit1,newdata=trainData4), trainData4$classe)
```
Our first model, with all available predictors, gives an accuracy of 96.1272% with a confidence interval of 95.8028% to 96.4332%. This is relatively accurate model to define base on accelerometer readings. But, before we attempt at making the model more accurate, let us explore the important variables for this model.


```r
vIS <- varImp(modelFit1)
vIn <- vIS$Overall
vNames <- rownames(vIS)
vNames <- vNames[order(vIn, decreasing=TRUE)]
```

Now, that we have a list of variables in the order of their importance, lets build models with top 10, 20 and 30 variables and observe the loss in accuracy.


```r
set.seed(999)
topTEN <- trainData4[,colnames(trainData4) %in% vNames[1:10]]
modelFit2 <- randomForest(trainData4$classe ~ ., data = topTEN, ntree = 1)
t10CM <- confusionMatrix(predict(modelFit2,newdata=topTEN), trainData4$classe)
```


```r
set.seed(999)
topTwenty <- trainData4[,colnames(trainData4) %in% vNames[1:20]]
modelFit3 <- randomForest(trainData4$classe ~ ., data = topTwenty, ntree = 1)
t20CM <- confusionMatrix(predict(modelFit3,newdata=topTwenty), trainData4$classe)
```


```r
set.seed(999)
topThirty <- trainData4[,colnames(trainData4) %in% vNames[1:30]]
modelFit4 <- randomForest(trainData4$classe ~ ., data = topThirty, ntree = 1)
t30CM <- confusionMatrix(predict(modelFit4,newdata=topThirty), trainData4$classe)
```

From the confusion matrix resutls, one can see that the model with 30 variables has an accuracy of 96.6436% with a confidence interval of 96.3398% and 96.9287%. We have reduced the model by almost 50% predictors while changing the accuracy by only -0.5164%.

At this point, we have two models under consideration,  
- Model 1 with all the available predictors and an in-sample accuracy of 96.1272%   
- Model 4 with all 30 of the 52 available predictors and an in-sample accuracy of 96.6436%

### Boosting the accuracy
Next we try to boost the accuracy of our two models by increasing the number of trees to 20 for each model. 


```r
set.seed(999)
modelFit1 <- randomForest(classe ~ ., data = trainData4, ntree = 20)
aCM <- confusionMatrix(predict(modelFit1,newdata=trainData4), trainData4$classe)
```

```r
set.seed(999)
topThirty <- trainData4[,colnames(trainData4) %in% vNames[1:30]]
modelFit4 <- randomForest(trainData4$classe ~ ., data = topThirty, ntree = 20)
t30CM <- confusionMatrix(predict(modelFit4,newdata=topThirty), trainData4$classe)
```

One would notice that the accuracy 100% & 100% is nearly equal for both the models under consideration, and that there has been a significant jump in accuracy with boosting. 

Before we move to validation, it might be worthwhile to perform a similar boosting to a model with 10 variables to access the accuracy.


```r
set.seed(999)
topTEN <- trainData4[,colnames(trainData4) %in% vNames[1:10]]
modelFit2 <- randomForest(trainData4$classe ~ ., data = topTEN, ntree = 20)
t10CM <- confusionMatrix(predict(modelFit2,newdata=topTEN), trainData4$classe)
```

With boosting, even this model seems to be fairly accurate at 99.9864%. If we can have a model with less predictors and almost an equivalent accuracy, it is wise to make a simpler, scalable, interpretable model requiring least amount of data/features.

Thus far, it seems like our model with just 10 variables is sufficiently accurate. So, now we will try out our 3 possible models on the validation set, and pick the one based on accuracy and simplicity.

## Validation

```r
aCMv <- confusionMatrix(predict(modelFit1,newdata=vTestData), vTestData$classe)
t30CMv <- confusionMatrix(predict(modelFit4,newdata=vTestData), vTestData$classe)
t10CMv <- confusionMatrix(predict(modelFit2,newdata=vTestData), vTestData$classe)
```

The validation results seem to be reflective of that there is minimal loss of accuracy, if we simplified the covariants from 52 to 30 to just 10, with out-of sample accuracy of 99.4494%, 99.2863% and 98.593% respectively.

It is also significant the in-sample versus out-of-sample accuracy change in a model with just 10 variable is much larger than that of a model with 30 or all 52 predictors.


#Applying Model to the Test Set
Given the numbers from our analysis, we will move forwar with a model with just 30 variables and apply it to the test set. At this point, we expect the out-of-sample error for the model to be 99.2863%

[Test Data][2] has also be made available to us by Coursera.


```r
setInternet2(TRUE)
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv" )
setInternet2(FALSE)
```

```
## [1] TRUE
```

```r
testing <- read.csv("pml-testing.csv")
```

```r
pclasse <- predict(modelFit2,newdata=testing)
pclasse
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

The results of pclasse were then written to text files using the below code and submitted for evaluation.


```r
pml_write_files <- function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(pclasse)
```

All predictions were made accurately for the submission using a model with just 30 variables.

#APPENDIX
## A | Summary of Predictor

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :2934   Min.   :1.32e+09     Min.   :   294      
##  1st Qu.: 4901   carlitos:2339   1st Qu.:1.32e+09     1st Qu.:252367      
##  Median : 9820   charles :2670   Median :1.32e+09     Median :500310      
##  Mean   : 9815   eurico  :2256   Mean   :1.32e+09     Mean   :501893      
##  3rd Qu.:14715   jeremy  :2588   3rd Qu.:1.32e+09     3rd Qu.:752314      
##  Max.   :19622   pedro   :1931   Max.   :1.32e+09     Max.   :998801      
##                                                                           
##           cvtd_timestamp new_window    num_window    roll_belt    
##  05/12/2011 11:24:1127   no :14404   Min.   :  1   Min.   :-28.9  
##  30/11/2011 17:11:1109   yes:  314   1st Qu.:223   1st Qu.:  1.1  
##  28/11/2011 14:14:1098               Median :425   Median :113.0  
##  05/12/2011 11:25:1081               Mean   :431   Mean   : 64.5  
##  02/12/2011 13:34:1036               3rd Qu.:643   3rd Qu.:123.0  
##  02/12/2011 14:58:1033               Max.   :864   Max.   :162.0  
##  (Other)         :8234                                            
##    pitch_belt        yaw_belt      total_accel_belt kurtosis_roll_belt
##  Min.   :-54.90   Min.   :-179.0   Min.   : 0.0              :14404   
##  1st Qu.:  1.73   1st Qu.: -88.3   1st Qu.: 3.0     #DIV/0!  :    9   
##  Median :  5.28   Median : -13.0   Median :17.0     -1.908453:    2   
##  Mean   :  0.24   Mean   : -10.9   Mean   :11.3     -0.016850:    1   
##  3rd Qu.: 14.60   3rd Qu.:  13.7   3rd Qu.:18.0     -0.021024:    1   
##  Max.   : 60.30   Max.   : 179.0   Max.   :29.0     -0.025513:    1   
##                                                     (Other)  :  300   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :14404            :14404              :14404   
##  #DIV/0!  :   23     #DIV/0!:  314     #DIV/0!  :    8   
##  47.000000:    4                       0.000000 :    3   
##  -0.684748:    3                       0.422463 :    2   
##  -1.750749:    3                       -0.003095:    1   
##  11.094417:    3                       -0.014020:    1   
##  (Other)  :  278                       (Other)  :  299   
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt   max_picth_belt 
##           :14404             :14404     Min.   :-94     Min.   : 3     
##  #DIV/0!  :   23      #DIV/0!:  314     1st Qu.:-88     1st Qu.: 5     
##  -2.156553:    3                        Median : -5     Median :18     
##  -3.072669:    3                        Mean   : -7     Mean   :13     
##  0.000000 :    3                        3rd Qu.: 14     3rd Qu.:19     
##  6.855655 :    3                        Max.   :180     Max.   :30     
##  (Other)  :  279                        NA's   :14404   NA's   :14404  
##   max_yaw_belt   min_roll_belt   min_pitch_belt   min_yaw_belt  
##         :14404   Min.   :-180    Min.   : 0             :14404  
##  -0.9   :   22   1st Qu.: -88    1st Qu.: 3      -0.9   :   22  
##  -1.1   :   22   Median :  -7    Median :17      -1.1   :   22  
##  -1.4   :   20   Mean   : -11    Mean   :11      -1.4   :   20  
##  -1.2   :   19   3rd Qu.:   3    3rd Qu.:17      -1.2   :   19  
##  -0.7   :   18   Max.   : 173    Max.   :23      -0.7   :   18  
##  (Other):  213   NA's   :14404   NA's   :14404   (Other):  213  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   :  0         Min.   : 0                  :14404     
##  1st Qu.:  0         1st Qu.: 1           #DIV/0!:    9     
##  Median :  1         Median : 1           0.00   :   10     
##  Mean   :  4         Mean   : 2           0.0000 :  295     
##  3rd Qu.:  2         3rd Qu.: 2                             
##  Max.   :360         Max.   :12                             
##  NA's   :14404       NA's   :14404                          
##  var_total_accel_belt avg_roll_belt   stddev_roll_belt var_roll_belt  
##  Min.   : 0           Min.   :-27     Min.   : 0       Min.   :  0    
##  1st Qu.: 0           1st Qu.:  1     1st Qu.: 0       1st Qu.:  0    
##  Median : 0           Median :116     Median : 0       Median :  0    
##  Mean   : 1           Mean   : 69     Mean   : 1       Mean   :  7    
##  3rd Qu.: 0           3rd Qu.:123     3rd Qu.: 1       3rd Qu.:  0    
##  Max.   :16           Max.   :157     Max.   :14       Max.   :201    
##  NA's   :14404        NA's   :14404   NA's   :14404    NA's   :14404  
##  avg_pitch_belt  stddev_pitch_belt var_pitch_belt   avg_yaw_belt  
##  Min.   :-51     Min.   :0         Min.   : 0      Min.   :-138   
##  1st Qu.:  2     1st Qu.:0         1st Qu.: 0      1st Qu.: -88   
##  Median :  5     Median :0         Median : 0      Median :  -6   
##  Mean   :  1     Mean   :1         Mean   : 1      Mean   :  -9   
##  3rd Qu.: 16     3rd Qu.:1         3rd Qu.: 0      3rd Qu.:   6   
##  Max.   : 60     Max.   :4         Max.   :16      Max.   : 174   
##  NA's   :14404   NA's   :14404     NA's   :14404   NA's   :14404  
##  stddev_yaw_belt  var_yaw_belt    gyros_belt_x      gyros_belt_y    
##  Min.   :  0     Min.   :    0   Min.   :-1.0400   Min.   :-0.6400  
##  1st Qu.:  0     1st Qu.:    0   1st Qu.:-0.0300   1st Qu.: 0.0000  
##  Median :  0     Median :    0   Median : 0.0300   Median : 0.0200  
##  Mean   :  1     Mean   :  138   Mean   :-0.0049   Mean   : 0.0399  
##  3rd Qu.:  1     3rd Qu.:    0   3rd Qu.: 0.1100   3rd Qu.: 0.1100  
##  Max.   :177     Max.   :31183   Max.   : 2.2200   Max.   : 0.6400  
##  NA's   :14404   NA's   :14404                                      
##   gyros_belt_z    accel_belt_x      accel_belt_y    accel_belt_z   
##  Min.   :-1.46   Min.   :-120.00   Min.   :-69.0   Min.   :-275.0  
##  1st Qu.:-0.20   1st Qu.: -21.00   1st Qu.:  3.0   1st Qu.:-162.0  
##  Median :-0.10   Median : -15.00   Median : 34.0   Median :-152.0  
##  Mean   :-0.13   Mean   :  -5.47   Mean   : 30.2   Mean   : -72.8  
##  3rd Qu.:-0.02   3rd Qu.:  -5.00   3rd Qu.: 61.0   3rd Qu.:  27.0  
##  Max.   : 1.62   Max.   :  83.00   Max.   :164.0   Max.   : 105.0  
##                                                                    
##  magnet_belt_x   magnet_belt_y magnet_belt_z     roll_arm     
##  Min.   :-52.0   Min.   :354   Min.   :-623   Min.   :-180.0  
##  1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375   1st Qu.: -31.9  
##  Median : 35.0   Median :601   Median :-319   Median :   0.0  
##  Mean   : 55.7   Mean   :594   Mean   :-346   Mean   :  17.6  
##  3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306   3rd Qu.:  77.0  
##  Max.   :481.0   Max.   :673   Max.   : 293   Max.   : 180.0  
##                                                               
##    pitch_arm         yaw_arm        total_accel_arm var_accel_arm  
##  Min.   :-88.80   Min.   :-180.00   Min.   : 1.0    Min.   :  0    
##  1st Qu.:-26.10   1st Qu.: -42.70   1st Qu.:17.0    1st Qu.: 11    
##  Median :  0.00   Median :   0.00   Median :27.0    Median : 41    
##  Mean   : -4.72   Mean   :  -0.79   Mean   :25.5    Mean   : 53    
##  3rd Qu.: 11.10   3rd Qu.:  44.70   3rd Qu.:33.0    3rd Qu.: 75    
##  Max.   : 88.50   Max.   : 180.00   Max.   :66.0    Max.   :253    
##                                                     NA's   :14404  
##   avg_roll_arm   stddev_roll_arm  var_roll_arm   avg_pitch_arm  
##  Min.   :-167    Min.   :  0     Min.   :    0   Min.   :-82    
##  1st Qu.: -37    1st Qu.:  1     1st Qu.:    2   1st Qu.:-22    
##  Median :   0    Median :  6     Median :   32   Median :  0    
##  Mean   :  13    Mean   : 11     Mean   :  459   Mean   : -5    
##  3rd Qu.:  74    3rd Qu.: 14     3rd Qu.:  207   3rd Qu.:  8    
##  Max.   : 161    Max.   :162     Max.   :26232   Max.   : 76    
##  NA's   :14404   NA's   :14404   NA's   :14404   NA's   :14404  
##  stddev_pitch_arm var_pitch_arm    avg_yaw_arm    stddev_yaw_arm 
##  Min.   : 0       Min.   :   0    Min.   :-173    Min.   :  0    
##  1st Qu.: 1       1st Qu.:   2    1st Qu.: -25    1st Qu.:  3    
##  Median : 8       Median :  68    Median :   0    Median : 17    
##  Mean   :10       Mean   : 200    Mean   :   4    Mean   : 22    
##  3rd Qu.:16       3rd Qu.: 269    3rd Qu.:  39    3rd Qu.: 36    
##  Max.   :43       Max.   :1885    Max.   : 152    Max.   :177    
##  NA's   :14404    NA's   :14404   NA's   :14404   NA's   :14404  
##   var_yaw_arm     gyros_arm_x      gyros_arm_y      gyros_arm_z    
##  Min.   :    0   Min.   :-6.370   Min.   :-3.440   Min.   :-2.330  
##  1st Qu.:    7   1st Qu.:-1.350   1st Qu.:-0.800   1st Qu.:-0.070  
##  Median :  276   Median : 0.080   Median :-0.260   Median : 0.250  
##  Mean   : 1007   Mean   : 0.049   Mean   :-0.263   Mean   : 0.271  
##  3rd Qu.: 1277   3rd Qu.: 1.570   3rd Qu.: 0.140   3rd Qu.: 0.720  
##  Max.   :31345   Max.   : 4.870   Max.   : 2.840   Max.   : 2.690  
##  NA's   :14404                                                     
##   accel_arm_x    accel_arm_y      accel_arm_z      magnet_arm_x 
##  Min.   :-383   Min.   :-318.0   Min.   :-636.0   Min.   :-584  
##  1st Qu.:-241   1st Qu.: -54.0   1st Qu.:-144.0   1st Qu.:-297  
##  Median : -43   Median :  14.0   Median : -48.0   Median : 296  
##  Mean   : -60   Mean   :  32.3   Mean   : -71.8   Mean   : 195  
##  3rd Qu.:  84   3rd Qu.: 138.0   3rd Qu.:  22.0   3rd Qu.: 642  
##  Max.   : 437   Max.   : 308.0   Max.   : 292.0   Max.   : 782  
##                                                                 
##   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm kurtosis_picth_arm
##  Min.   :-386   Min.   :-597           :14404            :14404    
##  1st Qu.: -11   1st Qu.: 129   #DIV/0! :   61    #DIV/0! :   61    
##  Median : 200   Median : 441   -0.02438:    1    -0.00484:    1    
##  Mean   : 156   Mean   : 305   -0.04190:    1    -0.01311:    1    
##  3rd Qu.: 323   3rd Qu.: 544   -0.05051:    1    -0.10385:    1    
##  Max.   : 583   Max.   : 694   -0.08050:    1    -0.11279:    1    
##                                (Other) :  249    (Other) :  249    
##  kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
##          :14404           :14404            :14404             :14404  
##  #DIV/0! :    7   #DIV/0! :   60    #DIV/0! :   61     #DIV/0! :    7  
##  0.55844 :    2   -0.00051:    1    -0.00184:    1     -1.62032:    2  
##  0.65132 :    2   -0.00696:    1    -0.02063:    1     -0.00562:    1  
##  -0.01548:    1   -0.01884:    1    -0.02652:    1     -0.01697:    1  
##  -0.01749:    1   -0.03484:    1    -0.02986:    1     -0.03455:    1  
##  (Other) :  301   (Other) :  250    (Other) :  249     (Other) :  302  
##   max_roll_arm   max_picth_arm    max_yaw_arm     min_roll_arm  
##  Min.   :-73     Min.   :-173    Min.   : 4      Min.   :-89    
##  1st Qu.:  0     1st Qu.:  -2    1st Qu.:29      1st Qu.:-42    
##  Median :  6     Median :  23    Median :34      Median :-22    
##  Mean   : 11     Mean   :  36    Mean   :35      Mean   :-22    
##  3rd Qu.: 28     3rd Qu.:  97    3rd Qu.:41      3rd Qu.:  0    
##  Max.   : 85     Max.   : 180    Max.   :65      Max.   : 66    
##  NA's   :14404   NA's   :14404   NA's   :14404   NA's   :14404  
##  min_pitch_arm    min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm
##  Min.   :-180    Min.   : 1      Min.   :  0        Min.   :  0        
##  1st Qu.: -70    1st Qu.: 7      1st Qu.:  5        1st Qu.: 10        
##  Median : -32    Median :12      Median : 29        Median : 54        
##  Mean   : -32    Mean   :14      Mean   : 33        Mean   : 68        
##  3rd Qu.:   0    3rd Qu.:19      3rd Qu.: 51        3rd Qu.:114        
##  Max.   : 152    Max.   :38      Max.   :120        Max.   :360        
##  NA's   :14404   NA's   :14404   NA's   :14404      NA's   :14404      
##  amplitude_yaw_arm roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##  Min.   : 0        Min.   :-153.5   Min.   :-149.6   Min.   :-148.77  
##  1st Qu.:13        1st Qu.: -18.9   1st Qu.: -40.9   1st Qu.: -77.60  
##  Median :22        Median :  48.4   Median : -21.1   Median :  -5.34  
##  Mean   :21        Mean   :  23.8   Mean   : -10.9   Mean   :   1.12  
##  3rd Qu.:28        3rd Qu.:  67.8   3rd Qu.:  17.1   3rd Qu.:  78.45  
##  Max.   :52        Max.   : 153.6   Max.   : 149.4   Max.   : 154.95  
##  NA's   :14404                                                        
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :14404                 :14404                  :14404        
##  #DIV/0!:    3          -0.5464:    2           #DIV/0!:  314        
##  -0.2583:    2          -0.9334:    2                                
##  -0.5855:    2          -2.0889:    2                                
##  -2.0889:    2          #DIV/0!:    2                                
##  -0.0115:    1          -0.0163:    1                                
##  (Other):  304          (Other):  305                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :14404                 :14404                  :14404        
##  -0.9324:    2          -0.7036:    2           #DIV/0!:  314        
##  #DIV/0!:    2          1.0326 :    2                                
##  -0.0096:    1          -0.0053:    1                                
##  -0.0393:    1          -0.0084:    1                                
##  -0.0430:    1          -0.0166:    1                                
##  (Other):  307          (Other):  307                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   :-70       Min.   :-113              :14404    Min.   :-150     
##  1st Qu.:-27       1st Qu.: -67       -0.6   :   15    1st Qu.: -60     
##  Median : 15       Median :  43       -0.8   :   14    Median : -44     
##  Mean   : 15       Mean   :  34       -0.3   :   13    Mean   : -41     
##  3rd Qu.: 51       3rd Qu.: 133       0.2    :   13    3rd Qu.: -23     
##  Max.   :130       Max.   : 155       -0.1   :   12    Max.   :  73     
##  NA's   :14404     NA's   :14404      (Other):  247    NA's   :14404    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   :-147              :14404    Min.   :  0            
##  1st Qu.: -92       -0.6   :   15    1st Qu.: 16            
##  Median : -63       -0.8   :   14    Median : 36            
##  Mean   : -32       -0.3   :   13    Mean   : 56            
##  3rd Qu.:  23       0.2    :   13    3rd Qu.: 79            
##  Max.   : 121       -0.1   :   12    Max.   :256            
##  NA's   :14404      (Other):  247    NA's   :14404          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
##  Min.   :  0                     :14404          Min.   : 0.0        
##  1st Qu.: 18              #DIV/0!:    3          1st Qu.: 4.0        
##  Median : 42              0.00   :  311          Median :10.0        
##  Mean   : 65                                     Mean   :13.7        
##  3rd Qu.: 96                                     3rd Qu.:20.0        
##  Max.   :271                                     Max.   :58.0        
##  NA's   :14404                                                       
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   :  0        Min.   :-129      Min.   :  0         
##  1st Qu.:  0        1st Qu.: -13      1st Qu.:  5         
##  Median :  1        Median :  49      Median : 13         
##  Mean   :  4        Mean   :  23      Mean   : 21         
##  3rd Qu.:  3        3rd Qu.:  64      3rd Qu.: 26         
##  Max.   :230        Max.   : 118      Max.   :124         
##  NA's   :14404      NA's   :14404     NA's   :14404       
##  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   :    0     Min.   :-70        Min.   : 0           
##  1st Qu.:   22     1st Qu.:-43        1st Qu.: 4           
##  Median :  158     Median :-20        Median : 8           
##  Mean   :  996     Mean   :-12        Mean   :13           
##  3rd Qu.:  690     3rd Qu.: 13        3rd Qu.:19           
##  Max.   :15321     Max.   : 94        Max.   :83           
##  NA's   :14404     NA's   :14404      NA's   :14404        
##  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
##  Min.   :   0       Min.   :-114     Min.   :  0         Min.   :    0   
##  1st Qu.:  13       1st Qu.: -77     1st Qu.:  4         1st Qu.:   18   
##  Median :  67       Median :   2     Median : 10         Median :  103   
##  Mean   : 370       Mean   :   1     Mean   : 16         Mean   :  574   
##  3rd Qu.: 351       3rd Qu.:  71     3rd Qu.: 23         3rd Qu.:  552   
##  Max.   :6836       Max.   : 135     Max.   :107         Max.   :11468   
##  NA's   :14404      NA's   :14404    NA's   :14404       NA's   :14404   
##  gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
##  Min.   :-204.00   Min.   :-2.10    Min.   : -2.4    Min.   :-419    
##  1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3    1st Qu.: -51    
##  Median :   0.13   Median : 0.03    Median : -0.1    Median :  -9    
##  Mean   :   0.16   Mean   : 0.05    Mean   : -0.1    Mean   : -29    
##  3rd Qu.:   0.35   3rd Qu.: 0.21    3rd Qu.:  0.0    3rd Qu.:  10    
##  Max.   :   2.22   Max.   :52.00    Max.   :317.0    Max.   : 234    
##                                                                      
##  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189.0   Min.   :-334.0   Min.   :-643      Min.   :-744     
##  1st Qu.:  -8.0   1st Qu.:-142.0   1st Qu.:-535      1st Qu.: 232     
##  Median :  42.5   Median :  -2.0   Median :-480      Median : 311     
##  Mean   :  52.9   Mean   : -39.3   Mean   :-331      Mean   : 223     
##  3rd Qu.: 112.0   3rd Qu.:  36.0   3rd Qu.:-308      3rd Qu.: 391     
##  Max.   : 310.0   Max.   : 318.0   Max.   : 592      Max.   : 633     
##                                                                       
##  magnet_dumbbell_z  roll_forearm     pitch_forearm     yaw_forearm    
##  Min.   :-262      Min.   :-180.00   Min.   :-72.50   Min.   :-180.0  
##  1st Qu.: -45      1st Qu.:  -0.58   1st Qu.:  0.00   1st Qu.: -68.3  
##  Median :  13      Median :  21.95   Median :  9.41   Median :   0.0  
##  Mean   :  45      Mean   :  34.19   Mean   : 10.82   Mean   :  19.3  
##  3rd Qu.:  94      3rd Qu.: 140.00   3rd Qu.: 28.50   3rd Qu.: 110.0  
##  Max.   : 452      Max.   : 180.00   Max.   : 89.80   Max.   : 180.0  
##                                                                       
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##         :14404                :14404                 :14404       
##  #DIV/0!:   64         #DIV/0!:   65          #DIV/0!:  314       
##  -0.9169:    2         -0.0073:    1                              
##  -0.0227:    1         -0.0442:    1                              
##  -0.0359:    1         -0.0489:    1                              
##  -0.0567:    1         -0.0891:    1                              
##  (Other):  245         (Other):  245                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##         :14404                :14404                 :14404       
##  #DIV/0!:   63         #DIV/0!:   65          #DIV/0!:  314       
##  -0.1912:    2         0.0000 :    2                              
##  -0.4126:    2         -0.0113:    1                              
##  -0.0004:    1         -0.0131:    1                              
##  -0.0013:    1         -0.0478:    1                              
##  (Other):  245         (Other):  244                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
##  Min.   :-64      Min.   :-149             :14404   Min.   :-67     
##  1st Qu.:  0      1st Qu.:   0      #DIV/0!:   64   1st Qu.: -6     
##  Median : 28      Median : 113      -1.2   :   26   Median :  0     
##  Mean   : 26      Mean   :  83      -1.3   :   21   Mean   :  0     
##  3rd Qu.: 46      3rd Qu.: 174      -1.4   :   20   3rd Qu.: 12     
##  Max.   : 88      Max.   : 180      -1.6   :   20   Max.   : 60     
##  NA's   :14404    NA's   :14404     (Other):  163   NA's   :14404   
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   :-180             :14404   Min.   :  0           
##  1st Qu.:-175      #DIV/0!:   64   1st Qu.:  2           
##  Median : -59      -1.2   :   26   Median : 19           
##  Mean   : -56      -1.3   :   21   Mean   : 26           
##  3rd Qu.:   0      -1.4   :   20   3rd Qu.: 41           
##  Max.   : 167      -1.6   :   20   Max.   :126           
##  NA's   :14404     (Other):  163   NA's   :14404         
##  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
##  Min.   :  0                    :14404         Min.   :  0.0      
##  1st Qu.:  2             #DIV/0!:   64         1st Qu.: 29.0      
##  Median : 85             0.00   :  250         Median : 36.0      
##  Mean   :139                                   Mean   : 34.7      
##  3rd Qu.:350                                   3rd Qu.: 41.0      
##  Max.   :360                                   Max.   :108.0      
##  NA's   :14404                                                    
##  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
##  Min.   :  0       Min.   :-177     Min.   :  0         Min.   :    0   
##  1st Qu.:  8       1st Qu.:  -1     1st Qu.:  0         1st Qu.:    0   
##  Median : 23       Median :  11     Median :  8         Median :   68   
##  Mean   : 35       Mean   :  32     Mean   : 42         Mean   : 5194   
##  3rd Qu.: 54       3rd Qu.: 104     3rd Qu.: 79         3rd Qu.: 6253   
##  Max.   :173       Max.   : 177     Max.   :179         Max.   :32102   
##  NA's   :14404     NA's   :14404    NA's   :14404       NA's   :14404   
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
##  Min.   :-65       Min.   : 0           Min.   :   0      Min.   :-155   
##  1st Qu.:  0       1st Qu.: 0           1st Qu.:   0      1st Qu.: -26   
##  Median : 12       Median : 6           Median :  34      Median :   0   
##  Mean   : 13       Mean   : 8           Mean   : 149      Mean   :  18   
##  3rd Qu.: 29       3rd Qu.:13           3rd Qu.: 173      3rd Qu.:  86   
##  Max.   : 70       Max.   :48           Max.   :2280      Max.   : 169   
##  NA's   :14404     NA's   :14404        NA's   :14404     NA's   :14404  
##  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x   gyros_forearm_y 
##  Min.   :  0        Min.   :    0   Min.   :-22.000   Min.   : -7.02  
##  1st Qu.:  1        1st Qu.:    0   1st Qu.: -0.220   1st Qu.: -1.48  
##  Median : 26        Median :  656   Median :  0.050   Median :  0.03  
##  Mean   : 45        Mean   : 4688   Mean   :  0.159   Mean   :  0.09  
##  3rd Qu.: 88        3rd Qu.: 7823   3rd Qu.:  0.580   3rd Qu.:  1.65  
##  Max.   :198        Max.   :39009   Max.   :  3.970   Max.   :311.00  
##  NA's   :14404      NA's   :14404                                     
##  gyros_forearm_z  accel_forearm_x  accel_forearm_y accel_forearm_z 
##  Min.   : -8.09   Min.   :-498.0   Min.   :-632    Min.   :-446.0  
##  1st Qu.: -0.18   1st Qu.:-180.0   1st Qu.:  57    1st Qu.:-182.0  
##  Median :  0.08   Median : -58.0   Median : 201    Median : -40.0  
##  Mean   :  0.16   Mean   : -63.1   Mean   : 164    Mean   : -55.8  
##  3rd Qu.:  0.49   3rd Qu.:  75.0   3rd Qu.: 312    3rd Qu.:  25.0  
##  Max.   :231.00   Max.   : 477.0   Max.   : 923    Max.   : 291.0  
##                                                                    
##  magnet_forearm_x magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280    Min.   :-896     Min.   :-966     A:4185  
##  1st Qu.: -618    1st Qu.:   9     1st Qu.: 204     B:2848  
##  Median : -385    Median : 589     Median : 512     C:2567  
##  Mean   : -315    Mean   : 380     Mean   : 398     D:2412  
##  3rd Qu.:  -79    3rd Qu.: 736     3rd Qu.: 654     E:2706  
##  Max.   :  663    Max.   :1480     Max.   :1090             
## 
```

## B | Function rmvNACols

```r
rmvNAcols <- function (ds, n, count=FALSE, pct=0.5) {
    cols <- ncol(ds)
    NAcol <- NULL
    if(count==FALSE){
        for(i in 1:cols){
            nNA <- nrow(ds[is.na(ds[,i]),])
            if((nNA/length(ds[,i])) > pct){
                NAcol <- c(NAcol, i)
            }
        }
    }
    
    if(count==TRUE){
        for(i in 1:cols) {
            if(nrow(ds[is.na(ds[,i]),]) > n){
                NAcol <- c(NAcol, i)
            }
        }
    }
    
    print(NAcol)
    ds[ , -c(NAcol)] 
    
}
```

---
[1]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
[2]: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
