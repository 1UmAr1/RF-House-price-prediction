head(t)
head(otest$SalePrice)
head(pp2)
head(t)
# using xgboost now
# first we need to convert our data into a matrix,  as xgb only takes numerical values
library(xgboost)
library(Matrix)
colnames(X_train)
trainm <- sparse.model.matrix(SalePrice ~. -77, data = X_train)
trainm <- sparse.model.matrix(SalePrice ~ .-77, data = X_train)
trainm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                                TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                                GarageCars + + BsmtUnfSF + GarageArea +
                                WoodDeckSF + Neighborhood + TotalBath +
                                GarageScore + TotalSF, data = X_train)
trainm
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
testm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                               TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                               GarageCars + + BsmtUnfSF + GarageArea +
                               WoodDeckSF + Neighborhood + TotalBath +
                               GarageScore + TotalSF, data = X_test)
X_test <- train1[test_index]
# Build X_train, X_test
X_train <- train1[train_index, ]
X_test <- train1[test_index]
X_test <- train1[test_index,]
testm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                               TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                               GarageCars + + BsmtUnfSF + GarageArea +
                               WoodDeckSF + Neighborhood + TotalBath +
                               GarageScore + TotalSF, data = X_test)
test_label <- test["SalePrice"]
test_label <- test[,"SalePrice"]
test_label <- X_test[,"SalePrice"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)
#parameters
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
watchlist <- list(train = train_matrix, test = test_matrix)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
?xgb.train
trainm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                                TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                                GarageCars + + BsmtUnfSF + GarageArea +
                                WoodDeckSF + Neighborhood + TotalBath +
                                GarageScore + TotalSF, data = X_train)
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
#parameters
nc <- length(unique(train_label))
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
xgb_params <- list("objective" = "gbtree",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
xgb_params <- list("objective" = "linear",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
#parameters
train_label
#parameters
str(train_label)
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
library(xgboost)
library(Matrix)
trainm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                                TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                                GarageCars + + BsmtUnfSF + GarageArea +
                                WoodDeckSF + Neighborhood + TotalBath +
                                GarageScore + TotalSF, data = X_train)
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
testm <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                               TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                               GarageCars + + BsmtUnfSF + GarageArea +
                               WoodDeckSF + Neighborhood + TotalBath +
                               GarageScore + TotalSF, data = X_test)
test_label <- X_test[,"SalePrice"]
test_matrix <- xgb.DMatrix(data = as.matrix(testm), label = test_label)
#parameters
str(train_label)
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 2000,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
demo()
y_train <- train1[train_index]
y_test <- train1[-train_index]
X_train <- train1[train_index, ]
colnames(train1)
X_test <- train1[test_index,]
y_train <- sample(1:nrow(X_train), 0.66 * nrow(X_train))
y_test <- setdiff(1:nrow(X_train), X_train)
t <- train1[test_index, c(77)]
X_train <- train1[train_index, ]
colnames(train1)
X_test <- train1[test_index,]
y_train <- sample(1:nrow(X_test), 0.66 * nrow(X_test))
y_test <- setdiff(1:nrow(X_test), X_test)
rdg <- cv.glmnet(x_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
library(glmnet)
rdg <- cv.glmnet(x_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
rdg <- cv.glmnet(X_train, X_test, type.measure = "mse", alpha = 0, family = "gaussian")
# Build X_train, X_test
X_train <- train1[train_index, ]
colnames(train1)
X_test <- train1[test_index,]
y_train <- sample(1:nrow(X_train), 0.66 * nrow(X_train))
y_test <- setdiff(1:nrow(X_test), X_test)
t <- train1[test_index, c(77)]
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
X_train <- train1[train_index, ]
colnames(train1)
X_test <- train1[test_index,]
y_train <- sample(1:nrow(X_train), 0.8 * nrow(X_train))
y_test <- setdiff(1:nrow(X_test), X_test)
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
# Build X_train, X_test
X_train <- train1[train_index, ]
colnames(train1)
X_test <- train1[test_index,]
y_train <- sample(1:nrow(train1), 0.8 * nrow(train1))
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
# train_rows <- sample(1:train, 0.66)
# train <- x[train_rows]
y_train <- as.matrix(y_train)
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "binomila")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "mgaussian")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "cox")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "multinomial")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "possian")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "poisson")
rdg <- cv.glmnet(X_train, X_test, type.measure = "mse", alpha = 0, family = "poisson")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "poisson")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "binomial")
rdg <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
alpha <- cv.glmnet(X_train, type.measure = "mse", alpha = 0, family = "gaussian")
alpha <- cv.glmnet(X_train, train1, type.measure = "mse", alpha = 0, family = "gaussian")
alpha <- cv.glmnet(X_train, y_train, type.measure = "mse", alpha = 0, family = "gaussian")
colnames(train1)
# Build X_train, X_test
X_train <- train1[train_index, -77]
y_train <- train1[train_index, c[77]]
y_train <- train1[train_index, c(77)]
y_train
colnames(X_test)
X_train <- train1[train_index, -77]
y_train <- train1[train_index, c(77)]
X_test <- train1[test_index, -77]
y_test <- train1[test_index, c(77)]
X_train <- train1[train_index, -77]
y_train <- train1[test_index, -77]
X_test <- train1[train_index, c(77)]
y_test <- train1[test_index, c(77)]
x_tr <- build.x(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = X_train, contrasts = F, sparse = T)
y_tr <- build.y(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = y_train) %>% as.integer() -1
x_val <- build.x(SalePrice, data = X_test) %>% as.integer() -1
# Fence, PoolQC, MiscFeature, and alley features have
# a lot of missing values so it is better to remove them
library(dplyr)
x_tr <- build.x(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = X_train, contrasts = F, sparse = T)
y_tr <- build.y(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = y_train) %>% as.integer() -1
x_val <- build.x(SalePrice, data = X_test) %>% as.integer() -1
library(xgboost)
library(Matrix)
x_tr <- build.x(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = X_train, contrasts = F, sparse = T)
y_tr <- build.y(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                  TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                  GarageCars + + BsmtUnfSF + GarageArea +
                  WoodDeckSF + Neighborhood + TotalBath +
                  GarageScore + TotalSF, data = y_train) %>% as.integer() -1
x_val <- build.x(SalePrice, data = X_test) %>% as.integer() -1
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = X_train)
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
y_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = y_train)
x_v <- sparse.model.matrix(SalePrice, data = X_train)
x_v <- sparse.model.matrix(SalePrice, data = X_train)
x_v <- sparse.model.matrix(SalePrice, data = X_test)
x_v <- sparse.model.matrix(data = X_test)
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = X_train)
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
y_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = y_train)
x_v <- X_test[,"SalePrice"]
y_v <- y_test[,"SalePrice"]
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = X_train)
train_label <- X_train[,"SalePrice"]
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
y_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = y_train)
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = X_train)
X_train <- train1[train_index]
y_train <- train1[test_index]
X_test <- train1[train_index, c(77)]
y_test <- train1[test_index, c(77)]
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, -77, data = X_train)
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = X_train)
X_train <- train1[train_index,]
y_train <- train1[test_index,]
X_test <- train1[train_index, c(77)]
y_test <- train1[test_index, c(77)]
x_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = X_train)
y_t <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = y_train)
train_matrix <- xgb.DMatrix(data = as.matrix(trainm), label = train_label)
train_label <- X_train[,"SalePrice"]
x_v <- X_test[,"SalePrice"]
y_v <- y_test[,"SalePrice"]
x_v <- X_test[,"SalePrice"]
test_matrix <- xgb.DMatrix(data = as.matrix(x_tm), label = train_label)
test_matrix <- xgb.DMatrix(data = as.matrix(x_t), label = train_label)
train_matrix <- xgb.DMatrix(data = as.matrix(x_t), label = train_label)
test_matrix <- xgb.DMatrix(data = as.matrix(y_t), label = train_label)
test_matrix <- xgb.DMatrix(data = as.matrix(y_t), label = train_label)
str(train_label)
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
test_matrix <- xgb.DMatrix(data = as.matrix(y_t), label = y_train[, "SalePrice"])
str(train_label)
nc <- length(unique(train_label))
nc
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nc)
watchlist <- list(train = train_matrix, test = test_matrix)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = X_t,
                      nrounds = 100,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = x_t,
                      nrounds = 100,
                      watchlist = watchlist)
# making model
bst_model <-xgb.train(params = xgb_params,
                      data = train_matrix,
                      nrounds = 100,
                      watchlist = watchlist)
# using xgboost
library(xgboost)
library(matrix)
library(Matrix)
X_test <- train1[test_index, ]
X_test <- train1[test_index, -77]
y_test <- train1[test_index, c(77)]
sg_train <- xgb.DMatrix(data = x_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
X_train1 <- as.matrix(X_train)
t_train <- as.matrix(y_train)
X_train <- train1[train_index, -77]
y_train <- train1[train_index, c(77)]
X_test <- train1[test_index, -77]
y_test <- train1[test_index, c(77)]
X_train1 <- as.matrix(X_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train, label = y_train)
X_train1 <- as.matrix(X_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
X_train1 <- as.matrix(X_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
X_train1 <- as.numeric.matrix(X_train)
t_train <- as.numeric.matrix(y_train)
X_train1 <- matrix(X_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
X_train1 <- matrix(X_train)
t_train <- as.numeric.matrix(y_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
library(Matrix)
X_train1 <- matrix(X_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
spt <- sparse.model.matrix(SalePrice, data = train1)
spt <- sparse.model.matrix(SalePrice ~, data = train1)
spt <- sparse.model.matrix(data = train1)
?sparse.model.matrix
spt <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = train1)
train_index <- sample(1:nrow(train1), 0.8 * nrow(train1))
test_index <- setdiff(1:nrow(train1), train_index)
# Build X_train, X_test
X_train <- train1[train_index, -77]
y_train <- train1[train_index, c(77)]
X_test <- train1[test_index, -77]
y_test <- train1[test_index, c(77)]
X_train1 <- matrix(X_train)
t_train <- as.matrix(y_train)
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
t_train <- y_train["SalePrice"]
sg_train <- xgb.DMatrix(data = X_train1, label = t_train)
sg_train <- xgb.DMatrix(spt, data = train1)
spr <-
  sg_train <- xgb.DMatrix(spr, data = train1)
spr <-
  sg_train <- xgb.DMatrix(spt, data = train1)
sg_train <- xgb.DMatrix(spt, label = t_train )
yy <- as.matrix(train1["SalePrice"])
sg_train <- xgb.DMatrix(spt, label = y)
sg_train <- xgb.DMatrix(spt, label = yy)
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 1, nround = 5, nthread = 3, objective = "gblinear" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, objective = "multi:softprob" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, objective = "regression" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, objective = "gbtree" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, objective = "linear" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, objective = "gb:linear" )
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, booster = "gbtree", objective = "reg:linear" )
spt <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = X_train)
# Build X_train, X_test
X_train <- train1[train_index, ]
spt <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = X_train)
yy <- as.matrix(train1["SalePrice"])
sg_train <- xgb.DMatrix(spt, label = yy)
sg_train <- xgb.DMatrix(spt, label = yy)
spt <- sparse.model.matrix(SalePrice ~ OverallQual + YearBuilt + YearRemodAdd + BsmtFinSF1 +
                             TotalBsmtSF + X1stFlrSF + GrLivArea + FullBath + TotRmsAbvGrd + Fireplaces +
                             GarageCars + + BsmtUnfSF + GarageArea +
                             WoodDeckSF + Neighborhood + TotalBath +
                             GarageScore + TotalSF, data = X_train)
yy <- as.matrix(X_train["SalePrice"])
sg_train <- xgb.DMatrix(spt, label = yy)
bo <- xgboost(sg_train, num_class = 4, max.depth = 5, eta = 0.1, nround = 5,
              nthread = 3, booster = "gbtree", objective = "reg:linear" )
# combining the test and train
data <- rbind(train, test.SalePrice)
Na <- sapply(data, function(x) sum(is.na(x)))
Na.sum <- data.frame(index = names(data), Mis <- Na)
Na.sum[Na.sum$Mis > 0,]
save.image("C:/Users/Um Ar/R Projects/New BO/House_RE_Final/.RData")
