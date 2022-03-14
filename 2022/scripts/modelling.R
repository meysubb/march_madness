library(tidyverse)
library(tidymodels)
library(recipeselectors)
library(stacks)


raw_train_data <- readRDS("tourney_raw_train_data.RDS")

pre_19 <- raw_train_data %>% filter(Season<2019) %>% ungroup()

post_19 <- raw_train_data %>% filter(Season>=2019) %>% ungroup()


colSums(is.na(pre_19))

# This Boruta Algorithm is super cool 
# John was talking about it. Read more about it here: https://www.andreaperlato.com/mlpost/feature-selection-using-boruta-algorithm/

outright_boruta = recipe(HWin ~ ., data = pre_19 %>% select(-c(HDiff,Season,DayNum,HTeamID,HScore,ATeamID,AScore,NumOT,NeutralFlag,loc))) %>%
  step_impute_knn(all_predictors(), neighbors = 3) %>%
  step_select_boruta(all_predictors(), outcome = "HWin")

prepped = prep(outright_boruta)

# This reccomends which variables to use
prepped[['steps']][[2]][['res']][1]$finalDecision[which(prepped[['steps']][[2]][['res']][1]$finalDecision == 'Confirmed')]

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
                      # i hand pick these next few. 
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
  recipe(HWin ~ ., data = train %>% select(-c(HDiff))) 

hoops_wflow <- 
  workflow() %>% 
  add_recipe(tournament_rec)

ctrl_grid <- control_stack_grid()


cv_folds = vfold_cv(train %>% mutate(HWin = as.factor(HWin)) %>% select(-c(HDiff)), strata = HWin)

doParallel::registerDoParallel()

# k nearest neighbors 
knn_model_spec <- nearest_neighbor(
  neighbors = tune(),
  weight_func = tune(),
  dist_power = tune()
) %>%
  set_engine("kknn") %>%
  set_mode("classification")

knn_flow <- 
  hoops_wflow %>% 
  add_model(knn_model_spec)

knn_param <- parameters(knn_wf) 

knn_res <- tune_bayes(
  knn_flow,
  resamples = cv_folds,
  param_info = knn_param,
  iter = 1000,
  metrics = metric_set(mn_log_loss),
  initial = 15,
  control =  control_bayes(
    parallel_over = "resamples",
    no_improve = 100,
    uncertain = 10,
    save_pred = F,
    save_workflow = F,
    time_limit = 600,
    verbose = T
  )
)

saveRDS(knn_res,"2022/knn_tune.RDS")

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


# rf_grid <- grid_random(mtry(c(1, 13)),
#                        trees(),
#                        min_n(),
#                        size = 100)
# rand_forest_res <-
#   tune_grid(
#     object = rand_forest_wflow,
#     resamples = cv_folds,
#     grid = 10,
#     control = ctrl_grid,
#     metrics = metric_set(
#       mn_log_loss
#     )
#   )

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
  iter = 1000,
  metrics = metric_set(mn_log_loss),
  initial = 15,
  control =  control_bayes(
    parallel_over = "resamples",
    no_improve = 100,
    uncertain = 10,
    save_pred = F,
    save_workflow = F,
    time_limit = 600,
    verbose = T
  )
)

saveRDS(rf_res,"2022/random_forest_tune.RDS")

# neural net - log loss is pretty high in the 0.6's, do not use this
# also learning
# nnet_spec <-
#   mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("nnet")
# 
# nnet_rec <-
#   tournament_rec %>%
#   step_normalize(all_predictors())
# 
# nnet_wflow <-
#   hoops_wflow %>%
#   add_model(nnet_spec)
# 
# nnet_res <-
#   tune_grid(
#     object = nnet_wflow,
#     resamples = cv_folds,
#     grid = 10,
#     control = ctrl_grid,
#     metrics = metric_set(
#       mn_log_loss
#     )
#   )

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
  iter = 1000,
  metrics = metric_set(
    mn_log_loss
  ),
  initial = 15,
  control = control_bayes(
    parallel_over = "resamples",
    no_improve = 100,
    uncertain = 10,
    save_pred = F,
    save_workflow = F,
    time_limit = 600,
    verbose = T
  )
)

saveRDS(xgb_tune,"2022/xgb_tune.RDS")

# stack models
model_st <- 
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(knn_res) %>% 
  add_candidates(rf_res) %>%
  add_candidates(xgb_tune) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()


readRDS(model_st,"2022/models_stacked.RDS")



# test <-
#   test %>% bind_cols(predict(model_st, .))
# test_pred <-
#   test %>%
#   bind_cols(predict(model_st, ., type = "prob"))
# 
# 
# yardstick::roc_auc(
#   test_pred,
#   truth = as.factor(HWin),
#   .pred_TRUE
# )
# 
