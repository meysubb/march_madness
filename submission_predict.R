library(tidyverse)
source("2022/scripts/utils.R")


seeds <- read_csv("2022/kaggle_data/MNCAATourneySeeds.csv") %>% 
  mutate(
    SeedNum = as.numeric(str_extract(Seed,"\\d+")),
    TeamID = as.character(TeamID)
  ) %>% select(-Seed) 

teams <- read_csv("2022/kaggle_data/MTeams.csv") %>% select(-FirstD1Season,-LastD1Season) %>% 
  mutate(TeamID=as.character(TeamID))

submission <- read_csv("2022/kaggle_data/MSampleSubmissionStage2.csv") %>% 
  tidyr::separate(ID,sep = "_",
                  into=c('Season',"HTeamID",
                         "ATeamID")) %>% mutate(Season=as.numeric(Season))

adv_ratings <- readRDS("2022/adv_rtgs.RDS") 
basic_ratings <- readRDS("2022/full_season_basic_rtgs.RDS") %>% unnest() 
massey_ratings <- readRDS("2022/mass_rtgs.RDS") %>% filter(Season>=2003) %>% 
  select(Season,TeamID,MAS,MOR,POM,SAG,WLK,DOL) %>% mutate(TeamID=as.character(TeamID))

## columns we care to keep 
metrics_selected <- c("SRS","RPI","MOV","TS","Colley","LRMC","Elo",
                      "MOR","POM","SAG","WLK","DOL",
                      # i hand pick these next few, lets see how it goes
                      "PTS_POSMade","PTS_POSAllowed","FGM3_POSMade","FGM3_POSAllowed")

res <- c("SeedDiff")

keep_cols = c(res, paste0(metrics_selected,'_H'), paste0(metrics_selected,'_A'))

submission_merged = submission  %>% inner_join(adv_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  inner_join(adv_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(basic_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  left_join(basic_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(massey_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  left_join(massey_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(seeds,by=c("HTeamID"="TeamID","Season")) %>% 
  left_join(seeds,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>% 
  mutate(
    SeedDiff = SeedNum_H - SeedNum_A
  ) %>% select(-SeedNum_H,-SeedNum_A) %>% 
  select(HTeamID,ATeamID,Season,keep_cols)

## from here we start the prediction 
xgb_model <- readRDS("2022/xgb_final_mod.RDS")

preds <- predict(xgb_model,new_data = submission_merged,type = "prob")

final_xgb_df <- cbind(submission_merged,preds) %>% 
  select(Season,HTeamID,ATeamID,Pred=.pred_TRUE) %>% 
  left_join(teams,by=c("HTeamID"="TeamID")) %>% 
  left_join(teams,by=c("ATeamID"="TeamID"),suffix=c("_H","_A")) %>% 
  adjust_matchup(.,"TCU","Seton Hall","TCU",0.7) %>% 
  adjust_matchup(.,"Ohio St","Loyola-Chicago","Loyola-Chicago",0.7) %>%
  adjust_matchup(.,"Marquette","North Carolina","Marquette",0.7) %>% 
  adjust_matchup(.,"Texas","Virginia Tech","Virginia Tech",0.65) %>% 
  adjust_matchup(.,"Creighton","San Diego St","Creighton",0.65) %>% 
  adjust_matchup(.,"Texas Tech","Alabama","Alabama",0.65) %>% 
  adjust_matchup(.,"LSU","Wisconsin","Wisconsin",0.65) %>% 
  adjust_matchup(.,"Houston","Illinois","Houston",0.65) %>% 
  adjust_matchup(.,"Houston","Wisconsin","Houston",0.65) %>% 
  adjust_matchup(.,"Tennessee","Villanova","Villanova",0.6) %>% 
  adjust_matchup(.,"Purdue","Kentucky","Purdue",0.75) %>% 
  adjust_matchup(.,"Kansas","Auburn","Auburn",0.6) %>% 
  select(-contains("TeamName")) %>% 
  unite(ID,Season:ATeamID,sep="_") 

# cap predictions

write_csv(final_xgb_df,"submission_stage2_xgb_adjusted.csv")

## stack predictions
rf_res <- readRDS("2022/random_forest_tune.RDS")
nnet_res <- readRDS("2022/neural_net.RDS")

library(stacks)
model_st <-
  # initialize the stack
  stacks() %>%
  # add candidate members
  add_candidates(rf_res) %>%
  add_candidates(nnet_res) %>%
  # determine how to combine their predictions
  blend_predictions() %>%
  # fit the candidates with nonzero stacking coefficients
  fit_members()

preds_stack <- predict(model_st,new_data=submission_merged,type="prob")

final_stack_df <- cbind(submission_merged,preds_stack) %>% 
  select(Season,HTeamID,ATeamID,Pred=.pred_TRUE) 

write_csv(
  final_stack_df %>% unite(ID, Season:ATeamID, sep = "_") ,
  "submission_stage2_stacks_unadjusted.csv"
)

adjusted_stack_df <- final_stack_df %>%
  left_join(teams,by=c("HTeamID"="TeamID")) %>% 
  left_join(teams,by=c("ATeamID"="TeamID"),suffix=c("_H","_A")) %>% 
  adjust_matchup(.,"Michigan St","Davidson","Michigan St",0.65) %>% 
  adjust_matchup(.,"TCU","Seton Hall","TCU",0.7) %>% 
  adjust_matchup(.,"Ohio St","Loyola-Chicago","Loyola-Chicago",0.7) %>%
  adjust_matchup(.,"Marquette","North Carolina","North Carolina",0.7) %>% 
  adjust_matchup(.,"Texas","Virginia Tech","Virginia Tech",0.65) %>% 
  adjust_matchup(.,"San Francisco","Murray St","San Francisco",0.7) %>% 
  adjust_matchup(.,"Arkansas","Connecticut","Arkansas",0.7) %>% 
  adjust_matchup(.,"Tennessee","Colorado St","Tennessee",0.8) %>% 
  adjust_matchup(.,"LSU","Wisconsin","LSU",0.65) %>% 
  adjust_matchup(.,"Tennessee","Villanova","Tennessee",0.6) %>% 
  adjust_matchup(.,"Baylor","UCLA","Baylor",0.75) %>% 
  adjust_matchup(.,"Purdue","Kentucky","Kentucky",0.75) %>% 
  adjust_matchup(.,"Kansas","Auburn","Kansas",0.6) %>% 
  adjust_matchup(.,"Arizona","Tennessee","Tennessee",0.6) %>% 
  select(-contains("TeamName")) %>% 
  unite(ID,Season:ATeamID,sep="_") 
  
write_csv(adjusted_stack_df,"submission_stage2_stacks_adjusted.csv")

# adjust some matchup scores and save predictions
adjust_matchup <- function(df,
                           team1,
                           team2,
                           winning_team_select,
                           win_prob) {
  ind1 <- df$TeamName_H %in% c(team1, team2)
  ind2 <- df$TeamName_A %in% c(team1, team2)
  complete_game <- which(ind1 & ind2)
  row <- df[complete_game, ]
  if (row$TeamName_H == winning_team_select) {
    df[complete_game, "Pred"] <- win_prob
  }
  if (row$TeamName_A == winning_team_select) {
    df[complete_game, "Pred"] <- 1 - win_prob
  }
  return(df)
}
  





# cap predictions

