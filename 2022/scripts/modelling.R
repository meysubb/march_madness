library(tidyverse)
library(tidymodels)
library(recipeselectors)
library(stacks)


raw_train_data <- readRDS("tourney_raw_train_data.RDS")

pre_19 <- raw_train_data %>% filter(Season<2016) %>% ungroup()

post_19 <- raw_train_data %>% filter(Season>=2016) %>% ungroup()


colSums(is.na(pre_19))

# This Boruta Algorithm is super cool 
# John was talking about it. Read more about it here: https://www.andreaperlato.com/mlpost/feature-selection-using-boruta-algorithm/

# outright_boruta = recipe(HWin ~ ., data = pre_19 %>% select(-c(HDiff,Season,DayNum,HTeamID,HScore,ATeamID,AScore,NumOT,NeutralFlag,loc))) %>%
#   step_impute_knn(all_predictors(), neighbors = 3) %>%
#   step_select_boruta(all_predictors(), outcome = "HWin")
#  
# prepped = prep(outright_boruta)
#  
# # This reccomends which variables to use
# prepped[['steps']][[2]][['res']][1]$finalDecision[which(prepped[['steps']][[2]][['res']][1]$finalDecision == 'Confirmed')]

## interestingly enough what I learnt is that doesn't really matter a lot of these ordinal values
## take into account a teams skill which might be measured by box score metrics.
# no_massey = pre_19 %>% select(-contains(c("MAS","MOR","POM","SAG","WLK","DOL","WLK")))
# 
# outright_boruta_no_mas = recipe(HWin ~ ., data = no_massey %>% select(-c(HDiff,Season,DayNum,HTeamID,HScore,ATeamID,AScore,NumOT,NeutralFlag,loc))) %>%
#   step_impute_knn(all_predictors(), neighbors = 3) %>%
#   step_select_boruta(all_predictors(), outcome = "HWin")
# 
# prep_no_mass = prep(outright_boruta_no_mas)
# 
# prep_no_mass[['steps']][[2]][['res']][1]$finalDecision[which(prep_no_mass[['steps']][[2]][['res']][1]$finalDecision == 'Confirmed')]


# I am going to add PPP and 3P rates as well even though it doesn't come up. Just doesn't make sense for me to drop that. 
# 
metrics_selected <- c("SRS","RPI","MOV","TS","Colley","LRMC","Elo",
                      "MOR","POM","SAG","WLK","DOL",
                      # i hand pick these next few, lets see how it goes
                      "PTS_POSMade","PTS_POSAllowed","FGM3_POSMade","FGM3_POSAllowed")

res <- c("HDiff","HWin","SeedDiff")

keep_cols = c(res, paste0(metrics_selected,'_H'), paste0(metrics_selected,'_A'))


# clf_impute = recipe(HWin ~ ., data = pre_19 %>% select(-c(HDiff))) %>%
#   step_knnimpute(all_predictors(), neighbors = 3)
# 
# prepped = prep(clf_impute)
# 
# data = juice(prepped)

set.seed(41)

train = pre_19 %>% select(keep_cols)
test = post_19 %>% select(keep_cols)

tournament_rec <- 
  recipe(HWin ~ ., data = train %>% select(-c(HDiff))) %>% 
  step_normalize(all_predictors()) %>% 
  step_impute_knn(all_predictors(), neighbors = 3)

hoops_wflow <- 
  workflow() %>% 
  add_recipe(tournament_rec)

# ctrl_grid <- control_stack_grid()

cv_folds = vfold_cv(train %>% mutate(HWin = as.factor(HWin)) %>% select(-c(HDiff)), strata = HWin)

# neural net - log loss is pretty high in the 0.6's, do not use this
# also learning
nnet_spec <-
  mlp(hidden_units = tune(), penalty = tune(),
      #dropout = tune(),
      epochs = tune()) %>%
  set_engine("nnet",trace=0) %>% 
  set_mode("classification")

nnet_wflow <-
  hoops_wflow %>%
  add_model(nnet_spec)

nnet_params <- 
  nnet_wflow %>% 
  parameters() %>% 
  update(hidden_units = hidden_units(c(1, 10)), 
         epochs = epochs(c(10, 1000)), 
         penalty = penalty(c(0.0000000001, 0.1)))

nnet_res <- tune_bayes(
  nnet_wflow,
  resamples = cv_folds,
  param_info = nnet_params,
  iter = 300,
  metrics = metric_set(
    mn_log_loss
  ),
  initial = 15,
  control = control_bayes(
    no_improve = 75,
    uncertain = 10,
    time_limit = 100,
    verbose = T,
    save_pred = T,
    save_workflow=T
  )
)

saveRDS(nnet_res,"2022/neural_net.RDS")

library(doParallel)
registerDoParallel()
cl <- makePSOCKcluster(6) # select the number of cores to parallelize the calcs across
registerDoParallel(cl)

# k nearest neighbors log loss is pretty high here too 
# knn_model_spec <- nearest_neighbor(
#   neighbors = tune(),
#   weight_func = tune(),
#   dist_power = tune()
# ) %>%
#   set_engine("kknn") %>%
#   set_mode("classification")
# 
# knn_flow <- 
#   hoops_wflow %>% 
#   add_model(knn_model_spec)
# 
# knn_param <- parameters(knn_flow) 
# 
# knn_res <- tune_bayes(
#   knn_flow,
#   resamples = cv_folds,
#   param_info = knn_param,
#   iter = 250,
#   metrics = metric_set(mn_log_loss),
#   initial = 15,
#   control =  control_bayes(
#     parallel_over = "resamples",
#     no_improve = 100,
#     uncertain = 10,
#     save_pred = F,
#     save_workflow = F,
#     time_limit = 600,
#     verbose = T
#   )
# )
# 
# saveRDS(knn_res,"2022/knn_tune.RDS")

# logistic regression -- lasso 
# this won't work for some reason. idk why
# lasso_model <- logistic_reg(penalty = tune(),
#                             mixture = 1) %>%
#   set_engine("glmnet") %>%
#   set_mode("classification")
# 
# lasso_wf <- hoops_wflow %>%
#   add_model(lasso_model)
# 
# lasso_param <- parameters(lasso_wf) %>% 
#   update(penalty= penalty(c(0,0.1)))
# 
# # 
# lasso_tune <- lasso_wf %>%
#   tune_bayes(
#     lasso_wf,
#     resamples = cv_folds,
#     param_info = lasso_param,
#     iter = 250,
#     metrics = metric_set(mn_log_loss),
#     initial = 5,
#     control =  control_bayes(
#       no_improve = 100,
#       uncertain = 10,
#       save_pred = F,
#       save_workflow = F,
#       time_limit = 600,
#       verbose = T
#     )
#   )
# #   
# saveRDS(lasso_tune,"2022/lasso_tune.RDS")

# random forest me
rand_forest_spec <- 
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    trees = tune(),
  ) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rand_forest_wflow <-
  hoops_wflow %>%
  add_model(rand_forest_spec)

rf_param <- 
  rand_forest_wflow %>% 
  parameters() %>% 
  update(mtry = mtry(c(5, 13)), 
         trees = trees(c(1, 2000)), 
         min_n = min_n(c(5, 30)))

rf_res <- tune_bayes(
  rand_forest_wflow,
  resamples = cv_folds,
  param_info = rf_param,
  iter = 300,
  metrics = metric_set(mn_log_loss),
  initial = 15,
  control =  control_bayes(
    parallel_over = "resamples",
    no_improve = 75,
    uncertain = 10,
    save_pred = T,
    save_workflow = T,
    time_limit = 100,
    verbose = T
  )
)

saveRDS(rf_res,"2022/random_forest_tune.RDS")

## XGB 
xgb = boost_tree(
  tree_depth = tune(),
  min_n = tune(),
  trees = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

xgb_params = parameters(
  tree_depth(),
  min_n(),
  trees(range = c(10, 2000)),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate()
)

xgb_wrkflw <- hoops_wflow %>% 
  add_model(xgb)

xgb_tune <- tune_bayes(
  xgb_wrkflw ,
  resamples = cv_folds,
  param_info = xgb_params,
  iter = 300,
  metrics = metric_set(
    mn_log_loss
  ),
  initial = 15,
  control = control_bayes(
    parallel_over = "resamples",
    no_improve = 75,
    uncertain = 10,
    save_pred = T,
    save_workflow = T,
    time_limit = 100,
    verbose = T
  )
)


saveRDS(xgb_tune,"2022/xgb_tune.RDS")


## Select Best Models
# rf_best <- select_best(rf_res,"mn_log_loss")
# rf_best
# 
# rf_final_mod <- 
#   rand_forest(
#     mtry = 5,
#     min_n = 30,
#     trees = 1610,
#   ) %>%
#   set_mode("classification") %>%
#   set_engine("ranger") %>% 
#   fit(HWin ~ ., data = train %>% mutate(HWin = as.factor(HWin)) %>% select(-c(HDiff)))
# 
# saveRDS(rf_final_mod,"rf_final_model.RDS")
# 
# nnet_best <- select_best(nnet_res,"mn_log_loss")
# nnet_best
# 
# nnet_final_mod <-
#   mlp(hidden_units = 8,
#       penalty = 1.22,
#       epochs = 530) %>%
#   set_engine("nnet", trace = 0) %>%
#   set_mode("classification") %>% 
#     fit(HWin ~ ., data = train %>% mutate(HWin = as.factor(HWin)) %>% select(-c(HDiff)))
# 
# saveRDS(nnet_final_mod,"nnet_final_model.RDS")
# 
# xgb_best <- select_best(xgb_tune, "mn_log_loss")
# xgb_best
# 
# full_df <- bind_rows(train,test)
# 
# xgb = boost_tree(
#   mtry = 21,
#   trees = 477,
#   min_n = 10,
#   tree_depth = 3,
#   learn_rate = 0.00611,
#   loss_reduction = 0.000000212,
#   sample_size = 0.993
# ) %>%
#   set_engine("xgboost") %>%
#   set_mode("classification") %>%
#   fit(HWin ~ ., data = full_df %>% mutate(HWin = as.factor(HWin)) %>% select(-c(HDiff)))
# 
# saveRDS(xgb,"xgb_final_mod.RDS")
# 
# 
# xprb = predict(xgb, test, type = "prob")
# rprb = predict(rf_final_mod,test,type="prob")
# nnetprb = predict(nnet_final_mod,test,type="prob")
# 
# libray(MLmetrics)
# print(LogLoss(xprb$.pred_TRUE,test$HWin))
# print(LogLoss(rprb$.pred_TRUE,test$HWin))
# don't use nnet when predicting a submission
# print(LogLoss(nnetprb$.pred_TRUE,test$HWin))

## note if you see any 0.99 predictions, send those to .95

# stack models
model_st <-
  # initialize the stack
  stacks() %>%
  # add candidate members
  # knn log loss eems quite high honestly but lets see.
  #add_candidates(knn_res) %>%
  # some problems with this
  #add_candidates(lasso_tune) %>%
  add_candidates(rf_res) %>%
  #add_candidates(xgb_tune) %>%
  add_candidates(nnet_res) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()

saveRDS(model_st,"2022/models_stacked.RDS")

stack_preds = predict(model_st,test,type="prob")
MLmetrics::LogLoss(stack_preds$.pred_TRUE,test$HWin)
# 
# 
# readRDS(model_st,"2022/models_stacked.RDS")
# 
