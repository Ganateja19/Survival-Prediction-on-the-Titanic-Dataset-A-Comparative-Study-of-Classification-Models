# Install and Load necessary libraries for building models
library(dplyr)    # For data manipulation
library(caret)    # For model training and evaluation
library(nnet)     # For multinomial logistic regression
library(MASS)     # For LDA and QDA
library(e1071)    # For Naive Bayes
library(class)    # For KNN
library(pROC)     # For AUC calculation
library(ggplot2)  # For Plotting 
library(ggcorrplot) #For correlation
library(car)


# Loading Titanic dataset
data<-read.csv("Titanic-Dataset.csv")


# Check for missing values
null_counts <- colSums(is.na(data))
print(null_counts)
total_nulls <- sum(is.na(data))
print(paste("Total null values in the dataset:", total_nulls))

skewness_value <- skewness(data$Age, na.rm = TRUE)
print(skewness_value)

boxplot(data$Age, main = "Boxplot of Age", ylab = "Age")

# Handle missing values
data <- data %>%
  mutate(Age = ifelse(is.na(Age), mean(Age, na.rm = TRUE), Age),
         Embarked = ifelse(Embarked == "", "S", Embarked)
         )

#Unique Counts
unique_counts <- sapply(data, function(x) length(unique(x)))
unique_counts

# Convert categorical variables to factors
data$Survived <- as.factor(data$Survived)
data$Sex <- as.factor(data$Sex)
data$Embarked <- as.factor(data$Embarked)
data$Pclass <- as.factor(data$Pclass)

#Convert int to numeric
data$SibSp <- as.numeric(data$SibSp)
data$Parch <- as.numeric(data$Parch)

# Bar plot of survival
survival_plot <- ggplot(data, aes(x = Survived, fill = Survived)) +
  geom_bar() +
  labs(title = "Distribution of Survival", x = "Survived", y = "Count") +
  theme_minimal()

#ggsave("survival_distribution.png", survival_plot, width = 6, height = 4)

survival_plot
table(data$Survived)


#Correlation
numeric_data <- data[, sapply(data, is.numeric)]

# Compute correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
print(cor_matrix)
ggcorrplot(cor_matrix, 
           method = "circle", 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           title = "Correlation Matrix of Numeric Features",
           colors = c("red", "white", "blue"))

# Select relevant columns
data <- data %>% dplyr::select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)



#Standardizing
nums <- sapply(data, is.numeric)
nums
data[nums] <- scale(data[nums])


# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$Survived, p = 0.8, list = FALSE)
train <- data[trainIndex, ]
test <- data[-trainIndex, ]
train_main <- train
test_main<-test


# Add interaction terms and second-order terms
train <- train %>%
  mutate(
        Age_Sex = Age*as.numeric(Sex),
         Fare_Pclass = Fare *as.numeric(Pclass),
         Age_Fare = Age*Fare,
         SibSp_Parch = SibSp*(Parch),
         Age_Pclass = Age*as.numeric(Pclass),
        Fare_SibSp = Fare*SibSp,
        Fare_Parch = Fare * Parch,
        Age_SibSp = Age*SibSp,
        Age_Parch = Age*Parch,
         Age2 = Age^2,
         Fare2 = Fare^2
        )

test <- test %>%
  mutate(
    Age_Sex = Age*as.numeric(Sex),
    Fare_Pclass = Fare *as.numeric(Pclass),
    Age_Fare = Age*Fare,
    SibSp_Parch = SibSp*(Parch),
    Age_Pclass = Age*as.numeric(Pclass),
    Fare_SibSp = Fare*SibSp,
    Fare_Parch = Fare * Parch,
    Age_SibSp = Age*SibSp,
    Age_Parch = Age*Parch,
    Age2 = Age^2,
    Fare2 = Fare^2,
    )

# Standardize numerical features
nums <- sapply(train, is.numeric)
nums
train[nums] <- scale(train[nums])
test[nums] <- scale(test[nums])



# Perform PCA on the data
pca_data <- model.matrix(Survived ~ . - 1, data = data)

pca_train <- prcomp(pca_data, center = TRUE, scale. = TRUE)

# Variance explained by each PC
variance_explained <- pca_train$sdev^2 / sum(pca_train$sdev^2)
print(variance_explained)

# Cumulative variance
cumulative_variance <- cumsum(variance_explained)
print(cumulative_variance)

# PC plot
plot(variance_explained, type = "b", main = "PC Plot", xlab = "Principal Component", ylab = "Variance Explained")

#With 85% of the variance explained, we are capturing most of the important patterns in the data
#while significantly reducing the dimensionality (from 10 original features to 6 PCs).

# Add PCA components to the dataset
pca_scores <- as.data.frame(pca_train$x[, 1:6])
pca_scores$Survived <- data$Survived


trainIndex <- createDataPartition(pca_scores$Survived, p = 0.8, list = FALSE)
train_pca <- pca_scores[trainIndex, ]
test_pca <- pca_scores[-trainIndex, ]


# Scatter plot of PCA components
pca_plot <- ggplot(train_pca, aes(x = PC1, y = PC2, color = Survived)) +
  geom_point(alpha = 0.6) +
  labs(title = "PCA Scatter Plot", x = "PC1", y = "PC2") +
  theme_minimal()
#ggsave("pca_scatter.png", pca_plot, width = 6, height = 4)

pca_plot


# Logistic Regression with main effects
logit_main <- glm(Survived~., data = train_main, family = binomial)
summary(logit_main)
vif(logit_main)
train_probs <- predict(logit_main, newdata = train_main, type = "response")
train_pred <- ifelse(train_probs > 0.5, 1, 0)
# Compare predictions with actual
main_train_accuracy <- mean(train_pred == train_main$Survived)

# Logistic Regression with interaction and second-order terms
logit_interaction <- glm(Survived ~ ., data = train, family = binomial)
summary(logit_interaction)
vif(logit_interaction)

train_probs <- predict(logit_interaction,newdata = train, type = "response")
train_pred <- ifelse(train_probs > 0.5, 1, 0)

# Compare predictions with actual
inter_train_accuracy <- mean(train_pred == train$Survived)


# Logistic Regression with PCA
logit_pca <- glm(Survived ~ ., data = train_pca, family = binomial)
summary(logit_pca)
vif(logit_pca)
train_probs <- predict(logit_pca, newdata = train_pca, type = "response")
train_pred <- ifelse(train_probs > 0.5, 1, 0)

# Compare predictions with actual
pca_train_accuracy <- mean(train_pred == train_pca$Survived)


print(main_train_accuracy)
print(inter_train_accuracy)
print(pca_train_accuracy)



# Naive Bayes with main effects
nb_main <- naiveBayes(Survived~., data = train_main)
summary(nb_main)
train_pred_nb <- predict(nb_main, newdata = train_main)
train_class_nb <- train_pred_nb  
main_accuracy_nb <- mean(train_class_nb == train_main$Survived)

# Naive Bayes with interaction and second-order terms
nb_interaction <- naiveBayes(Survived ~ ., data = train)
summary(nb_interaction)
train_pred_nb <- predict(nb_interaction, newdata = train)
train_class_nb <- train_pred_nb  
inter_accuracy_nb <- mean(train_class_nb == train$Survived)


# Naive Bayes with PCA
nb_pca <- naiveBayes(Survived ~ ., data = train_pca)
summary(nb_pca)
train_pred_nb <- predict(nb_pca, newdata = train_pca)
train_class_nb <- train_pred_nb  
pca_accuracy_nb <- mean(train_class_nb == train_pca$Survived)

print(main_accuracy_nb)
print(inter_accuracy_nb)
print(pca_accuracy_nb)


# LDA with main effects
lda_main <- lda(Survived~., data = train_main)
summary(lda_main)
train_pred_lda <- predict(lda_main, newdata = train_main)
train_class <- train_pred_lda$class
main_accuracy_lda <- mean(train_class == train_main$Survived)


# LDA with interaction and second-order terms
lda_interaction <- lda(Survived ~ ., data = train)
summary(lda_interaction)
lda_interaction$scaling
train_pred_lda <- predict(lda_interaction, newdata = train)
train_class <- train_pred_lda$class
inter_accuracy_lda <- mean(train_class == train$Survived)


# LDA with PCA
lda_pca <- lda(Survived ~ ., data = train_pca)
summary(lda_pca)
train_pred_lda <- predict(lda_pca, newdata = train_pca)
train_class <- train_pred_lda$class
pca_accuracy_lda <- mean(train_class == train_pca$Survived)

print(main_accuracy_lda)
print(inter_accuracy_lda)
print(pca_accuracy_lda)




# QDA with main effects
qda_main <- qda(Survived~., data = train_main)
summary(qda_main)
train_pred_qda <- predict(qda_main, newdata = train_main)
train_class <- train_pred_qda$class
main_accuracy_qda <- mean(train_class == train_main$Survived)


# QDA with interaction and second-order terms
qda_interaction <- qda(Survived ~ ., data = train)
summary(qda_interaction)
train_pred_qda <- predict(qda_interaction, newdata = train)
train_class <- train_pred_qda$class
inter_accuracy_qda <- mean(train_class == train$Survived)

# QDA with PCA
qda_pca <- qda(Survived ~ ., data = train_pca)
summary(qda_pca)
train_pred_qda <- predict(qda_pca, newdata = train_pca)
train_class <- train_pred_qda$class
pca_accuracy_qda <- mean(train_class == train_pca$Survived)

print(main_accuracy_qda)
print(inter_accuracy_qda)
print(pca_accuracy_qda)


#KNN

set.seed(123)
#Knn
tune_grid <- expand.grid(k = seq(1, 20, 2))  

knn_tune <- train(Survived ~ ., data = train, method = "knn",
                  tuneGrid = tune_grid, trControl = trainControl(method = "cv", number = 5))

best_k <- knn_tune$bestTune$k
print(best_k)

train_main_KNN <- model.matrix(~ .-Survived -1, data = train_main)
test_main_KNN <- model.matrix(~ .-Survived -1, data = test_main)

train_inter_KNN <- model.matrix(~ .-Survived -1, data = train)
test_inter_KNN <- model.matrix(~ . -Survived - 1, data = test)


# KNN with main effects with k=5
knn_main <- knn(train = train_main_KNN, 
                test = test_main_KNN, 
                cl = train_main$Survived, k = 5)
summary(knn_main)
# KNN with interaction and second-order terms
knn_interaction <- knn(train = train_inter_KNN, 
                       test = test_inter_KNN, 
                       cl = train$Survived, k = 5)
summary(knn_interaction)
# KNN with PCA
knn_pca <- knn(train = train_pca[, -ncol(train_pca)], 
               test = test_pca[, -ncol(test_pca)], 
               cl = train_pca$Survived, k = 5)
summary(knn_pca)




# Function to evaluate model performance
evaluate_model <- function(model, test_data, actual, model_type = "default") {
  if (model_type == "naiveBayes") {
    # For Naive Bayes, use type = "raw" to get probabilities
    print("Evaluating Naive Bayes model")
    predictions <- predict(model, test_data, type = "raw")[, 2]  
  } else if (model_type == "lda") {
    # For LDA, extract posterior probabilities for the positive class
    print("Evaluating LDA model")
    predictions <- predict(model, test_data)$posterior[, 2]  
  } else if (model_type == "qda") {
    # For QDA, extract posterior probabilities for the positive class
    print("Evaluating QDA model")
    predictions <- predict(model, test_data)$posterior[, 2]  
  } else if (model_type == "knn") {
    # For KNN, use the predicted class labels directly
    print("Evaluating KNN model")
    predictions <- as.numeric(model) - 1  
  } else {
    # For Logisitic regression, use type = "response" to get probabilities
    print("Evaluating Logistic model")
    predictions <- predict(model, test_data, type = "response")
  }
  # Convert probabilities to binary predictions (0 or 1)
  predicted_class <- ifelse(predictions > 0.5, 1, 0)
  
  # Confusion Matrix
  cm <- table(Actual = actual, Predicted = predicted_class)
  print("Confusion Matrix:")
  print(cm)
  
  TP <- cm["1", "1"]  
  TN <- cm["0", "0"]  
  FP <- cm["0", "1"]  
  FN <- cm["1", "0"]  
  
  # Calculate metrics
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  acc <- (TP + TN) / (TP + TN + FP + FN)
  # Print the results
  cat("Precision:", round(precision, 3), "\n")
  cat("Recall:", round(recall, 3), "\n")
  cat("F1 Score:", round(f1_score, 3), "\n")
  cat("Accuracy: ", round(acc, 3), "\n")
  # Calculate AUC
  roc_curve <- roc(actual, predictions)
  auc_value <- auc(roc_curve)
  return(auc_value)
}



# Evaluate Logistic Regression models
auc_logit_main <- evaluate_model(logit_main, test_main, as.numeric(test$Survived) - 1)
auc_logit_interaction <- evaluate_model(logit_interaction, test, as.numeric(test$Survived) - 1)
auc_logit_pca <- evaluate_model(logit_pca, test_pca, as.numeric(test_pca$Survived) - 1)

# Evaluate Naive Bayes models
auc_nb_main <- evaluate_model(nb_main, test_main, as.numeric(test$Survived) - 1, model_type = "naiveBayes")
auc_nb_interaction <- evaluate_model(nb_interaction, test, as.numeric(test$Survived) - 1, model_type = "naiveBayes")
auc_nb_pca <- evaluate_model(nb_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "naiveBayes")

# Evaluate LDA models
auc_lda_main <- evaluate_model(lda_main, test_main, as.numeric(test$Survived) - 1, model_type = "lda")
auc_lda_interaction <- evaluate_model(lda_interaction, test, as.numeric(test$Survived) - 1, model_type = "lda")
auc_lda_pca <- evaluate_model(lda_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "lda")

# Evaluate QDA models
auc_qda_main <- evaluate_model(qda_main, test_main, as.numeric(test$Survived) - 1, model_type = "qda")
auc_qda_interaction <- evaluate_model(qda_interaction, test, as.numeric(test$Survived) - 1, model_type = "qda")
auc_qda_pca <- evaluate_model(qda_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "qda")

# Evaluate KNN models
auc_knn_main <- evaluate_model(knn_main, test_main, as.numeric(test$Survived) - 1, model_type = "knn")
auc_knn_interaction <- evaluate_model(knn_interaction, test, as.numeric(test$Survived) - 1, model_type = "knn")
auc_knn_pca <- evaluate_model(knn_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "knn")

# Print AUC values
cat("Logistic Regression (Main Effects):", auc_logit_main, "\n")
cat("Logistic Regression (Interaction Terms):", auc_logit_interaction, "\n")
cat("Logistic Regression (PCA):", auc_logit_pca, "\n")
cat("Naive Bayes (Main Effects):", auc_nb_main, "\n")
cat("Naive Bayes (Interaction Terms):", auc_nb_interaction, "\n")
cat("Naive Bayes (PCA):", auc_nb_pca, "\n")
cat("LDA (Main Effects):", auc_lda_main, "\n")
cat("LDA (Interaction Terms):", auc_lda_interaction, "\n")
cat("LDA (PCA):", auc_lda_pca, "\n")
cat("QDA (Main Effects):", auc_qda_main, "\n")
cat("QDA (Interaction Terms):", auc_qda_interaction, "\n")
cat("QDA (PCA):", auc_qda_pca, "\n")
cat("KNN (Main Effects):", auc_knn_main, "\n")
cat("KNN (Interaction Terms):", auc_knn_interaction, "\n")
cat("KNN (PCA):", auc_knn_pca, "\n")









# Function to evaluate model performance and return ROC object
get_roc <- function(model, test_data, actual, model_type = "default") {
  if (model_type == "naiveBayes") {
    predictions <- predict(model, test_data, type = "raw")[, 2]  
  } else if (model_type == "lda") {
    predictions <- predict(model, test_data)$posterior[, 2]  
  } else if (model_type == "qda") {
    predictions <- predict(model, test_data)$posterior[, 2]  
  } else if (model_type == "knn") {
    predictions <- as.numeric(model) - 1  
  } else {
    predictions <- predict(model, test_data, type = "response")
  }
  roc(actual, predictions)
}

# Get ROC objects for all models
roc_logit_main <- get_roc(logit_main, test_main, as.numeric(test$Survived) - 1)
roc_logit_interaction <- get_roc(logit_interaction, test, as.numeric(test$Survived) - 1)
roc_logit_pca <- get_roc(logit_pca, test_pca, as.numeric(test_pca$Survived) - 1)

roc_nb_main <- get_roc(nb_main, test_main, as.numeric(test$Survived) - 1, model_type = "naiveBayes")
roc_nb_interaction <- get_roc(nb_interaction, test, as.numeric(test$Survived) - 1, model_type = "naiveBayes")
roc_nb_pca <- get_roc(nb_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "naiveBayes")

roc_lda_main <- get_roc(lda_main, test_main, as.numeric(test$Survived) - 1, model_type = "lda")
roc_lda_interaction <- get_roc(lda_interaction, test, as.numeric(test$Survived) - 1, model_type = "lda")
roc_lda_pca <- get_roc(lda_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "lda")

roc_qda_main <- get_roc(qda_main, test_main, as.numeric(test$Survived) - 1, model_type = "qda")
roc_qda_interaction <- get_roc(qda_interaction, test, as.numeric(test$Survived) - 1, model_type = "qda")
roc_qda_pca <- get_roc(qda_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "qda")

roc_knn_main <- get_roc(knn_main, test_main, as.numeric(test$Survived) - 1, model_type = "knn")
roc_knn_interaction <- get_roc(knn_interaction, test, as.numeric(test$Survived) - 1, model_type = "knn")
roc_knn_pca <- get_roc(knn_pca, test_pca, as.numeric(test_pca$Survived) - 1, model_type = "knn")

# Group ROC curves by type (Main Effects, Interaction Terms, PCA)
roc_main_effects <- list(
  roc_logit_main, roc_nb_main, roc_lda_main, roc_qda_main, roc_knn_main
)

roc_interaction_terms <- list(
  roc_logit_interaction, roc_nb_interaction, roc_lda_interaction, roc_qda_interaction, roc_knn_interaction
)

roc_pca <- list(
  roc_logit_pca, roc_nb_pca, roc_lda_pca, roc_qda_pca, roc_knn_pca
)

# Model names for the legend
model_names <- c(
  "Logistic Regression", "Naive Bayes", "LDA", "QDA", "KNN"
)

# Function to plot ROC curves for a given group
plot_roc_group <- function(roc_list, group_name, filename) {
  # AUC values for the legend
  auc_values <- sapply(roc_list, function(x) round(auc(x), 4))
  legend_labels <- paste(model_names, " (AUC = ", auc_values, ")", sep = "")
  
  # Plot ROC curves
  plot(roc_list[[1]], col = 1, main = paste("ROC Curves -", group_name), lwd = 2)
  for (i in 2:length(roc_list)) {
    lines(roc_list[[i]], col = i, lwd = 2)
  }
legend("bottomright", legend = legend_labels, col = 1:length(roc_list), lwd = 2, cex = 0.8)

# Save the plot
png(filename, width = 800, height = 600)
plot(roc_list[[1]], col = 1, main = paste("ROC Curves -", group_name), lwd = 2)
for (i in 2:length(roc_list)) {
  lines(roc_list[[i]], col = i, lwd = 2)
}
legend("bottomright", legend = legend_labels, col = 1:length(roc_list), lwd = 2, cex = 0.8)
dev.off()
}

# Plot and save ROC curves for each group
plot_roc_group(roc_main_effects, "Main Effects", "roc_main_effects.png")
plot_roc_group(roc_interaction_terms, "Interaction Terms", "roc_interaction_terms.png")
plot_roc_group(roc_pca, "PCA", "roc_pca.png")


#Printing summary table
auc_summary <- data.frame(
  Model = c("Logistic Regression", "Naive Bayes", "LDA", "QDA", "KNN"),
  Main = c(auc_logit_main, auc_nb_main, auc_lda_main, auc_qda_main, auc_knn_main),
  Interaction = c(auc_logit_interaction, auc_nb_interaction, auc_lda_interaction, auc_qda_interaction, auc_knn_interaction),
  PCA = c(auc_logit_pca, auc_nb_pca, auc_lda_pca, auc_qda_pca, auc_knn_pca)
)

# Print the summary
print(auc_summary)






## 10-Fold Cross Validation


# Combine main effects data
data_main <- rbind(train_main, test_main)

# Combine interaction and second-order terms data
data_inter <- rbind(train, test)

# Combine PCA data
data_pca <- rbind(train_pca, test_pca)




# Fix factor levels globally
data_main$Survived <- factor(data_main$Survived, levels = c("0", "1"), labels = c("No", "Yes"))
data_inter$Survived <- factor(data_inter$Survived, levels = c("0", "1"), labels = c("No", "Yes"))
data_pca$Survived <- factor(data_pca$Survived, levels = c("0", "1"), labels = c("No", "Yes"))

table(data_main$Survived)
str(data_main$Survived)
set.seed(123)
cv_auc <- function(data, model_method, model_type = "default") {
  folds <- createFolds(data$Survived, k = 10, list = TRUE, returnTrain = FALSE)
  aucs <- numeric(10)
  
  for (i in 1:10) {
    test_idx <- folds[[i]]
    train_data <- data[-test_idx, ]
    test_data <- data[test_idx, ]
    
    if (model_type == "knn") {
      train_x <- model.matrix(~ . - Survived - 1, data = train_data)
      test_x <- model.matrix(~ . - Survived - 1, data = test_data)
      pred <- knn(train = train_x, test = test_x, cl = train_data$Survived, k = 5)
      prob <- as.numeric(pred) - 1
      
    } else if (model_method == "nb") {
      model <- train(
        Survived ~ ., data = train_data, method = "nb",
        trControl = trainControl(method = "none", classProbs = TRUE),
        tuneGrid = expand.grid(fL = 1, usekernel = FALSE, adjust = 1)
      )
      prob <- predict(model, test_data, type = "prob")[, 2]
      
    } else if (model_method %in% c("lda", "qda")) {
      model <- train(
        Survived ~ ., data = train_data, method = model_method,
        trControl = trainControl(method = "none", classProbs = TRUE)
      )
      prob <- predict(model, test_data, type = "prob")[, 2]
      
    } else {
      model <- train(
        Survived ~ ., data = train_data, method = model_method,
        trControl = trainControl(method = "none"),
        family = if (model_method == "glm") binomial else NULL
      )
      prob <- predict(model, test_data, type = "prob")[, 2]
    }
    
    actual <- as.numeric(test_data$Survived) - 1
    aucs[i] <- auc(actual, prob)
  }
  
  mean(aucs)
}



# Main effects
auc_logit_cv_main <- cv_auc(data_main, "glm")
auc_nb_cv_main <- cv_auc(data_main, "nb", model_type = "nb")
auc_lda_cv_main <- cv_auc(data_main, "lda", model_type = "lda")
auc_qda_cv_main <- cv_auc(data_main, "qda", model_type = "qda")
auc_knn_cv_main <- cv_auc(data_main, "knn", model_type = "knn")

# Interaction terms
auc_logit_cv_inter <- cv_auc(data_inter, "glm")
auc_nb_cv_inter <- cv_auc(data_inter, "nb", model_type = "nb")
auc_lda_cv_inter <- cv_auc(data_inter, "lda", model_type = "lda")
auc_qda_cv_inter <- cv_auc(data_inter, "qda", model_type = "qda")
auc_knn_cv_inter <- cv_auc(data_inter, "knn", model_type = "knn")

# PCA
auc_logit_cv_pca <- cv_auc(data_pca, "glm")
auc_nb_cv_pca <- cv_auc(data_pca, "nb", model_type = "nb")
auc_lda_cv_pca <- cv_auc(data_pca, "lda", model_type = "lda")
auc_qda_cv_pca <- cv_auc(data_pca, "qda", model_type = "qda")
auc_knn_cv_pca <- cv_auc(data_pca, "knn", model_type = "knn")






cv_auc_summary <- data.frame(
  Model = c("Logistic Regression", "Naive Bayes", "LDA", "QDA", "KNN"),
  Main_CV_AUC = c(auc_logit_cv_main, auc_nb_cv_main, auc_lda_cv_main, auc_qda_cv_main, auc_knn_cv_main),
  Interaction_CV_AUC = c(auc_logit_cv_inter, auc_nb_cv_inter, auc_lda_cv_inter, auc_qda_cv_inter, auc_knn_cv_inter),
  PCA_CV_AUC = c(auc_logit_cv_pca, auc_nb_cv_pca, auc_lda_cv_pca, auc_qda_cv_pca, auc_knn_cv_pca)
)

print(cv_auc_summary)





