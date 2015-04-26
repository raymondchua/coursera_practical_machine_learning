activityTrain = read.csv("pml-training.csv")
activityTest = read.csv("pml-testing.csv")

sum(complete.cases(activityTrain))

#remove columns with NA values
activityTrain = activityTrain[,colSums(is.na(activityTrain)) == 0]
activityTest = activityTest[,colSums(is.na(activityTest)) == 0]

#remove column X, columns with Time Stamps and columns windows 
drops = c("X", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
activityTrain = activityTrain[,!(names(activityTrain) %in% drops)]
activityTest = activityTest[,!(names(activityTest) %in% drops)]

#convert all columns except classe to numeric
classe <- activityTrain$classe
activityTrain <- activityTrain[, sapply(activityTrain, is.numeric)]
activityTrain$classe <- classe
activityTest <- activityTest[, sapply(activityTest, is.numeric)]


#Split the data

library(caTools)
set.seed(3000)
spl = sample.split(activityTrain$classe, SplitRatio = 0.7)
Train = subset(activityTrain, spl==TRUE)
Test = subset(activityTrain, spl==FALSE)


library(caret)

library(randomForest)

library(corrplot)

library(e1071)





# Install rpart library
library(rpart)
library(rpart.plot)

# CART model
cartTree = rpart(classe ~ ., data = Train, method="class", minbucket=100)

# Make Predictions using the CART Model
PredictCART = predict(cartTree, newdata = Test, type = "class")

# Confusion Matrix using CART Model
confusionMatrixCart = confusionMatrix(Test$classe, PredictCART)
AccuracyCART = confusionMatrixCart$overall[1]

missClass = function(values, prediction) {
  sum(prediction != values)/length(values)
}
errRate = missClass(Test$classe, PredictCART)

# Correlation Matrix Visualization
corrPlot <- cor(Train[, -length(names(Train))])
corrplot(corrPlot, method="color")

#Classification Tree Visulation
prp(cartTree)



# Install randomForest package

library(randomForest)

#randomForest Model
trainForest = randomForest(classe ~ . , data = Train, ntree=200, nodesize=25)
predictForest = predict(trainForest, newdata = Test)

# Confusion Matrix using Random Forest Model
confusionMatrixRF = confusionMatrix(Test$classe, predictForest)
AccuracyRF = confusionMatrixRF$overall[1]

#random Forest Model 2
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=Train, method="rf", trControl=controlRf, ntree=200)
predictForest2 = predict(modelRf, newdata = Test)

confusionMatrixRF2 = confusionMatrix(Test$classe, predictForest2)
AccuracyRF2 = confusionMatrixRF2$overall[1]


