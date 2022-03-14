library(tidyverse)
source("2022/scripts/utils.R")


seeds <- read_csv("2022/kaggle_data/MNCAATourneySeeds.csv") %>% 
  mutate(
    SeedNum = as.numeric(str_extract(Seed,"\\d+"))
  ) %>% select(-Seed)

tourney_games <- read_csv("2022/kaggle_data/MNCAATourneyCompactResults.csv")
# ensure all of them are N
table(tourney_games$WLoc)

tourney_games_refac <- tourney_games %>% filter(Season>2002) %>% group_by(Season) %>% nest() %>% 
  mutate(
    data = purrr::map(data,refactor_game_dataframe)
  ) %>% unnest()

tourney_games_refac_flipped <- tourney_games_refac 
tourney_games_refac_flipped$HTeamID <- tourney_games_refac$ATeamID
tourney_games_refac_flipped$ATeamID <- tourney_games_refac$HTeamID
tourney_games_refac_flipped$HScore <- tourney_games_refac$AScore
tourney_games_refac_flipped$AScore <- tourney_games_refac$HScore

clean_tournament_games <- bind_rows(tourney_games_refac,tourney_games_refac_flipped) %>% 
  mutate(
    HDiff = HScore - AScore
  )

adv_ratings <- readRDS("2022/adv_rtgs.RDS") %>% mutate(TeamID=as.numeric(TeamID))
basic_ratings <- readRDS("2022/full_season_basic_rtgs.RDS") %>% unnest() %>% mutate(TeamID=as.numeric(TeamID))
massey_ratings <- readRDS("2022/mass_rtgs.RDS") %>% mutate(TeamID=as.numeric(TeamID)) %>% filter(Season>=2003) %>% 
  select(Season,TeamID,MAS,MOR,POM,SAG,WLK,DOL)
colSums(is.na(massey_ratings))

tournament_merged = clean_tournament_games  %>% inner_join(adv_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  inner_join(adv_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(basic_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  left_join(basic_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(massey_ratings,by=c('HTeamID'="TeamID","Season")) %>% 
  left_join(massey_ratings,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>%
  left_join(seeds,by=c("HTeamID"="TeamID","Season")) %>% 
  left_join(seeds,by=c("ATeamID"="TeamID","Season"),suffix=c("_H","_A")) %>% 
  mutate(
    HWin = HDiff > 0 ,
    SeedDiff = SeedNum_H - SeedNum_A
  ) %>% select(-SeedNum_H,-SeedNum_A)

saveRDS(tournament_merged,"tourney_raw_train_data.RDS")

