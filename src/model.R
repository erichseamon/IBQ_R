

library(dplyr)
library(lubridate)
library(tidyr)
library(data.table)
library(caret)
library(doParallel)
library(plyr)


ibq <- read.csv("/mnt/lfs2/erichs/git/IBQ_R/data/IBQ-R.csv")

ibq <- na.omit(ibq)
ibq <- subset(ibq, infantgender != -99999)
ibq <- subset(ibq, infantage_ibq != -99999)


#EDA

bwplot(researcher ~ infantage_ibq, data = ibq)
bwplot(infantgender ~ infantage_ibq, data = ibq)
bwplot(infantrace ~ infantage_ibq, data = ibq)

bwplot(AgeGrp ~ fear, data = ibq)
bwplot(AgeGrp ~ hp, data = ibq)
bwplot(AgeGrp ~ sooth, data = ibq)
bwplot(AgeGrp ~ cud, data = ibq)
bwplot(AgeGrp ~ vr, data = ibq)

bwplot(fear ~ infantage_ibq, data = ibq)

bwplot(infantgender ~ fear, data = ibq)
bwplot(infantgender ~ hp, data = ibq)
bwplot(infantgender ~ sooth, data = ibq)
bwplot(infantgender ~ cud, data = ibq)
bwplot(infantgender ~ vr, data = ibq)


#EDA


ibq$AgeGrp <- as.factor(ibq$AgeGrp)
ibq$AgeGrp <- revalue(ibq$AgeGrp, c("1"="one", "2"="two", "3"="three"))

ibq <- subset(ibq, AgeGrp == "three")

ibq$infantgender <- as.factor(ibq$infantgender)
levels(ibq$infantgender) <- c('male', 'female')

set.seed(103)
inTraining <- createDataPartition(ibq$infantgender, p = .7, list = FALSE)
training <- ibq[ inTraining,]
testing  <- ibq[-inTraining,]

fmla <- as.formula(paste("infantgender ~ ", paste(colnames(ibq[c(6,7,8,9,10,11,12,13,14,15,16,17,18,19)]), collapse= "+")))
predictors <- colnames(ibq[c(6,7,8,9,10,11,12,13,14,15,16,17,18,19)])

#fit <- caret::train(fmla, data=training, method="lm", trControl = train_control )

#train_control<- trainControl(method="LOOCV", number=10, savePredictions = "final", repeats = 3, returnResamp='all')
#train_control<- trainControl(method="LOOCV")

#fit <- caret::train(fmla, data=exposure_individual, method="lm", trControl = train_control )

train_control<- trainControl(method="repeatedcv", savePredictions = "final", classProbs = T, returnResamp='all')

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

fit_rf <- caret::train(fmla, data=training, method="rf", trControl = train_control)
fit_lda <- caret::train(fmla, data=training, method="lda", trControl = train_control)
fit_glm <- caret::train(fmla, data=training, method="glm", trControl = train_control)
fit_knn <- caret::train(fmla, data=training, method="knn",trControl = train_control)
fit_nb <- caret::train(fmla, data=training, method="nb", trControl = train_control)
fit_svm <- caret::train(fmla, data=training, method="svmRadial", trControl = train_control)
fit_cart <- caret::train(fmla, data=training, method="rpart", trControl = train_control)
fit_c50 <- caret::train(fmla, data=training, method="C5.0", trControl = train_control)
fit_treebag <- caret::train(fmla, data=training, method="treebag", trControl = train_control)
fit_gbm <- caret::train(fmla, data=training, method="gbm", trControl = train_control)
fit_adabag <- caret::train(fmla, data=training, method="AdaBag", trControl = train_control)



## When you are done:
stopCluster(cl)

#AgeGrp

results <- resamples(list(lda=fit_lda,
                          svm=fit_svm, knn=fit_knn, nb=fit_nb, cart=fit_cart, c50=fit_c50,
                          bagging=fit_treebag, rf=fit_rf, gbm=fit_gbm, adabag=fit_adabag))
# Table comparison
summary(results)

# boxplot comparison
bwplot(results)
# Dot-plot comparison
dotplot(results)


modellist <- c("svm", "rf", "treebag", "cart", "lda", "nb", "c50", "knn", "gbm")
leng <- 1:length(modellist)  
txt <- NULL
j <- 1
for (i in modellist) {
  modelnumber <- leng[j]
  library("ROCR")
  ### CONSTRUCTING ROC AUC PLOT:
  # Get the posteriors as a dataframe.
  predictions <-predict(object = eval(parse(text=paste("fit_", i, sep=""))), testing[,predictors], type = "prob")
  predictions.posteriors <- as.data.frame(predictions)
  
  # Evaluate the model
  pred <- prediction(predictions.posteriors[,2], testing[,3])
  roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
  auc.train <- performance(pred, measure = "auc")
  auc.train <- auc.train@y.values
  # Plot
  
  if(modelnumber == 1) {
    plot(roc.perf, col = j)  
    #abline(a=0, b= 1)
    #text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))
  } else{
    plot(roc.perf, col = j, add = TRUE)
    #abline(a=0, b= 1)
    txt[j] <- paste(i, " AUC = ", round(auc.train[[1]],3), sep = "")
  }
  j <- j+1
}

legend(.01,.99, txt)


#confusion matrix results


testing$pred_svm<-predict(object = fit_svm,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_svm)

testing$pred_rf<-predict(object = fit_rf,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_rf)


testing$pred_lda<-predict(object = fit_lda,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_lda)


testing$pred_gbm<-predict(object = fit_gbm,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_gbm)



#Predicting the out of fold prediction probabilities for training data
training$OOF_pred_glm<-fit_glm$pred$female[order(fit_glm$pred$rowIndex)]
training$OOF_pred_lda<-fit_lda$pred$female[order(fit_lda$pred$rowIndex)]
training$OOF_pred_svm<-fit_svm$pred$female[order(fit_svm$pred$rowIndex)]

#Predicting probabilities for the test data
testing$OOF_pred_glm<-predict(fit_glm,testing[predictors],type='prob')$female
testing$OOF_pred_lda<-predict(fit_lda,testing[predictors],type='prob')$female
testing$OOF_pred_svm<-predict(fit_svm,testing[predictors],type='prob')$female

#Predictors for top layer models 
predictors_top<-c('OOF_pred_glm','OOF_pred_lda','OOF_pred_svm') 

#GBM as top layer model 

model_gbm<- train(training[,predictors_top],training[,"infantgender"],method='gbm',trControl=train_control,tuneLength=3)
model_gbm<- train(infantgender ~ OOF_pred_glm + OOF_pred_lda +OOF_pred_svm,training,method='gbm',trControl=train_control,tuneLength=3)

# 
# actual <- testing$infantrace
# 
# actual_year <- testing$year
# obs_year <- training$year
# names.use <- names(testing)[(names(testing) %in% c(model_rfe$variables$var[1:8]))]
# testing_predictors <- testing[names.use]
# predicted <- unname(predict(fit, testing_predictors))
# 
# caret::RMSE(pred = fit$finalModel$fitted.values, obs = training$yield)
# caret::R2(pred = fit$finalModel$fitted.values, obs = training$yield)
# 
# caret::RMSE(pred = predicted, obs = testing$yield)
# caret::R2(pred = predicted, obs = testing$yield)
# 
# plot(actual_year, actual, ylim=c(3,7.5), xlab = "year", las=3)
# lines(actual_year, actual, col="black")
# points(actual_year, predicted, col="red")
# lines(actual_year, predicted, col="red")
# 
# plot(obs_year, fit$finalModel$fitted.values, ylim=c(3,7.5), xlab = "year", las=3, col="red")
# lines(obs_year, fit$finalModel$fitted.values, col="red")
# points(obs_year, training$yield, col="black", xlab = "year", las=3)
# lines(obs_year, training$yield, col="black")






#infantgender




results <- resamples(list(lda=fit_lda, logistic=fit_glm,
                          svm=fit_svm, knn=fit_knn, nb=fit_nb, cart=fit_cart, c50=fit_c50,
                          bagging=fit_treebag, rf=fit_rf, gbm=fit_gbm, adabag=fit_adabag))
# Table comparison
summary(results)

# boxplot comparison
bwplot(results)
# Dot-plot comparison
dotplot(results)


modellist <- c("svm", "rf", "treebag", "cart", "glm", "lda", "nb", "c50", "knn", "gbm")
leng <- 1:length(modellist)  
txt <- NULL
j <- 1
for (i in modellist) {
modelnumber <- leng[j]
library("ROCR")
### CONSTRUCTING ROC AUC PLOT:
# Get the posteriors as a dataframe.
predictions <-predict(object = eval(parse(text=paste("fit_", i, sep=""))), testing[,predictors], type = "prob")
predictions.posteriors <- as.data.frame(predictions)

# Evaluate the model
pred <- prediction(predictions.posteriors[,2], testing[,3])
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot

if(modelnumber == 1) {
plot(roc.perf, col = j)  
  #abline(a=0, b= 1)
  #text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))
} else{
plot(roc.perf, col = j, add = TRUE)
#abline(a=0, b= 1)
 txt[j] <- paste(i, " AUC = ", round(auc.train[[1]],3), sep = "")
}
j <- j+1
}

legend(.01,.99, txt)


#confusion matrix results


testing$pred_svm<-predict(object = fit_svm,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_svm)

testing$pred_rf<-predict(object = fit_rf,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_rf)


testing$pred_lda<-predict(object = fit_lda,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_lda)


testing$pred_glm<-predict(object = fit_glm,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_glm)


testing$pred_gbm<-predict(object = fit_gbm,testing[,predictors])
confusionMatrix(testing$infantgender,testing$pred_gbm)



#Predicting the out of fold prediction probabilities for training data
training$OOF_pred_glm<-fit_glm$pred$female[order(fit_glm$pred$rowIndex)]
training$OOF_pred_lda<-fit_lda$pred$female[order(fit_lda$pred$rowIndex)]
training$OOF_pred_svm<-fit_svm$pred$female[order(fit_svm$pred$rowIndex)]

#Predicting probabilities for the test data
testing$OOF_pred_glm<-predict(fit_glm,testing[predictors],type='prob')$female
testing$OOF_pred_lda<-predict(fit_lda,testing[predictors],type='prob')$female
testing$OOF_pred_svm<-predict(fit_svm,testing[predictors],type='prob')$female

#Predictors for top layer models 
predictors_top<-c('OOF_pred_glm','OOF_pred_lda','OOF_pred_svm') 

#GBM as top layer model 

model_gbm<- train(training[,predictors_top],training[,"infantgender"],method='gbm',trControl=train_control,tuneLength=3)
model_gbm<- train(infantgender ~ OOF_pred_glm + OOF_pred_lda +OOF_pred_svm,training,method='gbm',trControl=train_control,tuneLength=3)

# 
# actual <- testing$infantrace
# 
# actual_year <- testing$year
# obs_year <- training$year
# names.use <- names(testing)[(names(testing) %in% c(model_rfe$variables$var[1:8]))]
# testing_predictors <- testing[names.use]
# predicted <- unname(predict(fit, testing_predictors))
# 
# caret::RMSE(pred = fit$finalModel$fitted.values, obs = training$yield)
# caret::R2(pred = fit$finalModel$fitted.values, obs = training$yield)
# 
# caret::RMSE(pred = predicted, obs = testing$yield)
# caret::R2(pred = predicted, obs = testing$yield)
# 
# plot(actual_year, actual, ylim=c(3,7.5), xlab = "year", las=3)
# lines(actual_year, actual, col="black")
# points(actual_year, predicted, col="red")
# lines(actual_year, predicted, col="red")
# 
# plot(obs_year, fit$finalModel$fitted.values, ylim=c(3,7.5), xlab = "year", las=3, col="red")
# lines(obs_year, fit$finalModel$fitted.values, col="red")
# points(obs_year, training$yield, col="black", xlab = "year", las=3)
# lines(obs_year, training$yield, col="black")