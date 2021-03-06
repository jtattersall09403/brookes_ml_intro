# --- kNN experiment ----

# Get required packages
source('./R/00 Requirements.R')

# Get functions to read MNIST data
source('./R/00 Read MNIST script.R')

# Load data
train <- load_image_file('Data/train_images')
test <- load_image_file('Data/test_images')
train$y <- load_label_file('Data/train_labels')
test$y <- load_label_file('Data/test_labels')

# Get pca model
pca <- readRDS('./R/Models/pca.rds')

# Calculate proportion of variance explained by PCA
ev <- pca$sdev**2
desc <- rbind(
  SD = sqrt(ev),
  Proportion = ev/sum(ev),
  Cumulative = cumsum(ev)/sum(ev))

# Get number of components needed to explain more than 80% of the variation
c <- (min(which(desc[3,] > 0.8)))

# ------ Preparation --------

# Convert to single data frame for caret. Convert outcome variable to categorical, and remove variables with zero variances
dataTrain <- data.frame(digit = factor(train$y), train$x) %>% 
  sample_n(2000) %>%
  select(-nearZeroVar(.))

dataTest <- data.frame(digit = factor(test$y), test$x)

# Set up preprocessing. K-nn requires that predictors are normalised or scaled
preProc <- c("center", "scale")

# ---------- Setting up parameter tuning ----------

# Cross validation
fitControl <- trainControl(method = "repeatedcv",
                           savePredictions = T,
                           search = "random",
                           number = 5,
                           repeats = 3,
                           allowParallel = TRUE,
                           verboseIter = TRUE)

# Restrict to the faster models, add linear options
mdls <- list('knn'='knn')

# Define colours for plotting
pal <- wesanderson::wes_palette(name = "GrandBudapest", n = length(mdls), type = "discrete")
names(pal) <- mdls

# ---------- Model training ---------

set.seed(1)

# Train models
tune <- lapply(mdls, function(m){
  
  # Record progress
  print(paste("Training ", m))
  
  # Train model
  train(form= digit ~ .,
        data = dataTrain,
        method = m,
        metric = "Accuracy", # Accuracy is a suitable error measurement, as our classification problem is well balanced
        trControl = fitControl,
        preProcess = preProc,
        tuneLength = 20,
        na.action = na.exclude)
})

plot(tune$knn)


# -------- Get model accuracy metrics -------

# Define function to get model metrics
getRes <- function(i){
  name <- names(tune)[i]
  res <- tune[[i]]$results
  df <- res[2:ncol(res)]
  model <- apply(res,1,function(r) paste(r[1:(match('Accuracy', names(res)) - 1)],collapse = '-')) %>% 
    paste(name,.,sep='-')
  cbind.data.frame(model,df,name=name[[1]],stringsAsFactors =F)
}

# Combine into data frame
CVres <- plyr::ldply(1:length(tune),getRes)

# Rank by accuracy and kappa
CVres$rank <- rank(rank(1-CVres$Accuracy)+rank(1-CVres$Kappa),ties.method = 'first')

# Round for display and view
CVres[2:5] <- round(CVres[2:5],3)
CVres[order(CVres$rank),] %>% View

# ------- Plot model accuracy ----------

# Get predicted vs observed values for training data
# Set up function
getObsPred <- function(i){
  # i <- 2
  bst <- tune[[i]]$bestTune
  preds <- tune[[i]]$pred
  preds$name <- names(tune)[i]
  preds$model <- paste(bst,collapse = '-') %>% paste(names(tune)[i],.,sep='-')
  
  ii <- lapply(1:length(bst),function(p){
    preds[names(bst)[p]]==as.character(bst[p][[1]])
  })
  if(length(bst)>1) data.frame(ii) %>% apply(.,1,all) -> ii else unlist(ii) ->ii
  preds[ii,-which(names(preds)%in%names(bst))]
}

# Run on tune object
predObs <- plyr::ldply(1:length(tune),getObsPred)

# Get top models in each category
topmodels <- CVres %>% group_by(name) %>% filter(rank==min(rank))

# Get list of names
topmodels.lst <- topmodels$name[order(topmodels$rank)]
names(topmodels.lst) <- topmodels$model[order(topmodels$rank)]

# Convert model type to factor
resdf <- CVres %>%
  mutate(plot.score = Accuracy-AccuracySD) %>%
  group_by(name) %>%
  mutate(model_order = row_number(desc(plot.score))) %>%
  ungroup

# Cross validation plot 1
p1 <- ggplot(filter(resdf, model_order < 21),aes(x=reorder(model, plot.score),color=model))+
      geom_errorbar(aes(ymin=Accuracy-AccuracySD, ymax=Accuracy+AccuracySD),size=1)+
      geom_point(aes(y=Accuracy))+
      scale_color_viridis(discrete = TRUE, begin = 0.8, end = 0)+
      expand_limits(y = c(0, 1)) +
      coord_flip()+
      theme_bw()+
      xlab('')+
      labs(title='Accuracy metrics for top 10 models in each category') +
      theme(plot.title = element_text(hjust = -1)) +
      theme(legend.position='none')

p2 <- ggplot(filter(resdf, model_order < 21), aes(x=reorder(model, plot.score),color=model))+
      geom_errorbar(aes(ymin=Kappa-KappaSD, ymax=Kappa+KappaSD),size=1)+
      geom_point(aes(y=Kappa))+
      scale_color_viridis(discrete = TRUE, begin = 0.8, end = 0)+
      expand_limits(y = c(0, 1)) +
      coord_flip()+
      theme_bw()+
      xlab('')+
      labs(title = ' ') +
      theme(legend.position='none')

# Display
gridExtra::grid.arrange(p2, p1, ncol=2)

# Save
saveRDS(p1, './Plots/accuracy_plot.rds')
saveRDS(p2, './Plots/kappa_plot.rds')
saveRDS(tune, './R/Models/knn tune.rds')

# ---- Test set validation ----

dataTest <- dataTest[, names(dataTest) %in% names(dataTrain)]

# Add predictions to test set
dataTest$pred <- predict(tune$knn, newdata = dataTest)

# Check accuracy
sum(dataTest$pred == dataTest$digit)/nrow(dataTest)

# Save
saveRDS(dataTest, './Data/test_subset.rds')

# Load
dataTest <- readRDS('./Data/test_subset.rds')

# Plotting code sourced from MachLearn github repository
# Plot 1
p1 <- dataTest %>%
  group_by(pred, digit) %>% 
  summarise(n=n()) %>% 
  ggplot(aes(fill = pred))+
  geom_raster(aes(x=pred,y=digit,alpha=n))+
  geom_text(aes(x=pred,y=digit,label=n, alpha = n))+
  coord_equal()+
  scale_fill_viridis(discrete = TRUE) +
  theme_bw()+
  xlab('Predicted')+
  ylab('Observed')+
  labs(caption = "Test set predictions",
       tite = "Number of predictions in each class") +
  theme(legend.position='none')

# Plot 2
p2 <- dataTest %>%
  group_by(pred, digit) %>% 
  summarise(n=n()) %>% 
  group_by(pred) %>%
  mutate(pct = 100*round(n/sum(n), 2)) %>%
  ggplot(aes(fill = pred))+
  geom_raster(aes(x=pred,y=digit,alpha=pct))+
  geom_text(aes(x=pred,y=digit,label=pct, alpha = pct))+
  coord_equal()+
  scale_fill_viridis(discrete = TRUE) +
  theme_bw()+
  xlab('Predicted')+
  ylab('Observed')+
  labs(caption = "Test set predictions",
       title = "Percentage of predictions in each class") +
  theme(legend.position='none')

# Save plots
saveRDS(p1, './Plots/testset_plot_n.rds')
saveRDS(p2, './Plots/testset_plot_pct.rds')


# ------------ Dimension reduced KNN ---------

# ------ Preparation --------

# Get principal component scores for train data
train_pca <- predict(pca, newdata = train$x)

# Convert to single data frame for caret. Convert outcome variable to categorical, 
# and remove variables with zero variances. Select a random subset due to computational
# limitations
dataTrain <- data.frame(digit = factor(train$y), train_pca[,1:c]) %>%
  sample_n(2000) %>%
  select(-nearZeroVar(.))

# Do the same for the test data
test_pca <- predict(pca, newdata = test$x)
dataTest <- data.frame(digit = factor(test$y), test_pca[,1:c])

# Set up preprocessing. K-nn requires that predictors are normalised or scaled
preProc <- c("center", "scale")

# ---------- Setting up parameter tuning ----------

# Cross validation
fitControl <- trainControl(method = "repeatedcv",
                           savePredictions = T,
                           search = "random",
                           number = 5,
                           repeats = 3,
                           allowParallel = TRUE,
                           verboseIter = TRUE)

# Restrict to the faster models, add linear options
mdls <- list('knn'='knn')

# Define colours for plotting
pal <- wesanderson::wes_palette(name = "GrandBudapest", n = length(mdls), type = "discrete")
names(pal) <- mdls

# ---------- Model training ---------

set.seed(1)

# Train models
tune.2 <- lapply(mdls, function(m){
  
  # Record progress
  print(paste("Training ", m))
  
  # Train model
  train(form= digit ~ .,
        data = dataTrain,
        method = m,
        metric = "Accuracy", # Accuracy is a suitable error measurement, as our classification problem is well balanced
        trControl = fitControl,
        preProcess = preProc,
        tuneLength = 20,
        na.action = na.exclude)
})

plot(tune$knn)


# -------- Get model accuracy metrics -------

# Define function to get model metrics
getRes <- function(i){
  name <- names(tune.2)[i]
  res <- tune.2[[i]]$results
  df <- res[2:ncol(res)]
  model <- apply(res,1,function(r) paste(r[1:(match('Accuracy', names(res)) - 1)],collapse = '-')) %>% 
    paste(name,.,sep='-')
  cbind.data.frame(model,df,name=name[[1]],stringsAsFactors =F)
}

# Combine into data frame
CVres <- plyr::ldply(1:length(tune.2),getRes)

# Rank by accuracy and kappa
CVres$rank <- rank(rank(1-CVres$Accuracy)+rank(1-CVres$Kappa),ties.method = 'first')

# Round for display and view
CVres[2:5] <- round(CVres[2:5],3)
CVres[order(CVres$rank),] %>% View

# ------- Plot model accuracy ----------

# Get predicted vs observed values for training data
# Set up function
getObsPred <- function(i){
  # i <- 2
  bst <- tune.2[[i]]$bestTune
  preds <- tune.2[[i]]$pred
  preds$name <- names(tune.2)[i]
  preds$model <- paste(bst,collapse = '-') %>% paste(names(tune.2)[i],.,sep='-')
  
  ii <- lapply(1:length(bst),function(p){
    preds[names(bst)[p]]==as.character(bst[p][[1]])
  })
  if(length(bst)>1) data.frame(ii) %>% apply(.,1,all) -> ii else unlist(ii) ->ii
  preds[ii,-which(names(preds)%in%names(bst))]
}

# Run on tune object
predObs <- plyr::ldply(1:length(tune.2),getObsPred)

# Get top models in each category
topmodels <- CVres %>% group_by(name) %>% filter(rank==min(rank))

# Get list of names
topmodels.lst <- topmodels$name[order(topmodels$rank)]
names(topmodels.lst) <- topmodels$model[order(topmodels$rank)]

# Convert model type to factor
resdf <- CVres %>%
  mutate(plot.score = Accuracy-AccuracySD) %>%
  group_by(name) %>%
  mutate(model_order = row_number(desc(plot.score))) %>%
  ungroup

# Cross validation plot 1
p3 <- ggplot(filter(resdf, model_order < 21),aes(x=reorder(model, plot.score),color=model))+
  geom_errorbar(aes(ymin=Accuracy-AccuracySD, ymax=Accuracy+AccuracySD),size=1)+
  geom_point(aes(y=Accuracy))+
  scale_color_viridis(discrete = TRUE, begin = 0.8, end = 0)+
  expand_limits(y = c(0, 1)) +
  coord_flip()+
  theme_bw()+
  xlab('')+
  labs(title='Accuracy metrics for top 10 models in each category') +
  theme(plot.title = element_text(hjust = -1)) +
  theme(legend.position='none')

p4 <- ggplot(filter(resdf, model_order < 21), aes(x=reorder(model, plot.score),color=model))+
  geom_errorbar(aes(ymin=Kappa-KappaSD, ymax=Kappa+KappaSD),size=1)+
  geom_point(aes(y=Kappa))+
  scale_color_viridis(discrete = TRUE, begin = 0.8, end = 0)+
  expand_limits(y = c(0, 1)) +
  coord_flip()+
  theme_bw()+
  xlab('')+
  labs(title = ' ') +
  theme(legend.position='none')

# Display
gridExtra::grid.arrange(p4, p3, ncol=2)

# Save
saveRDS(p3, './Plots/accuracy_pca_plot.rds')
saveRDS(p4, './Plots/kappa_pca_plot.rds')
saveRDS(tune.2, './R/Models/knn pca tune.rds')

# ---- Test set validation ----

dataTest <- dataTest[, names(dataTest) %in% names(dataTrain)]

# Add predictions to test set
dataTest$pred <- predict(tune.2$knn, newdata = dataTest)

# Check accuracy
sum(dataTest$pred == dataTest$digit)/nrow(dataTest)

# Save
saveRDS(dataTest, './Data/test_pca_subset.rds')

# Load
dataTest <- readRDS('./Data/test_pca_subset.rds')

# Plotting code sourced from MachLearn github repository
# Plot 1
p1 <- dataTest %>%
  group_by(pred, digit) %>% 
  summarise(n=n()) %>% 
  ggplot(aes(fill = pred))+
  geom_raster(aes(x=pred,y=digit,alpha=n))+
  geom_text(aes(x=pred,y=digit,label=n, alpha = n))+
  coord_equal()+
  scale_fill_viridis(discrete = TRUE) +
  theme_bw()+
  xlab('Predicted')+
  ylab('Observed')+
  labs(caption = "Test set predictions",
       tite = "Number of predictions in each class") +
  theme(legend.position='none')

# Plot 2
p2 <- dataTest %>%
  group_by(pred, digit) %>% 
  summarise(n=n()) %>% 
  group_by(pred) %>%
  mutate(pct = 100*round(n/sum(n), 2)) %>%
  ggplot(aes(fill = pred))+
  geom_raster(aes(x=pred,y=digit,alpha=pct))+
  geom_text(aes(x=pred,y=digit,label=pct, alpha = pct))+
  coord_equal()+
  scale_fill_viridis(discrete = TRUE) +
  theme_bw()+
  xlab('Predicted')+
  ylab('Observed')+
  labs(caption = "Test set predictions",
       title = "Percentage of predictions in each class") +
  theme(legend.position='none')

# Save plots
saveRDS(p1, './Plots/testset_pca_plot_n.rds')
saveRDS(p2, './Plots/testset_pca_plot_pct.rds')
