source("2022/scripts/utils.R")
library(tidyverse)
library(reticulate)
library(lsei)
source_python("2022/scripts/create_team_value_basic.py")

team_names <- read_csv("2022/kaggle_data/MTeams.csv")
team_conf <- read_csv("2022/kaggle_data/MTeamConferences.csv")



calc_hfa <- function(game_dataframe){
  # drop neutral court games 
  games <- game_dataframe %>% filter(!NeutralFlag)
  x_df <- games %>% 
    mutate(
      HDiff = ifelse(NumOT>0,0,HScore - AScore)
    ) %>% select(HTeamID,ATeamID,HDiff)
  
  y_df <- games %>%  mutate(
    HWin = AScore>HScore
  ) %>% select(HTeamID,ATeamID,HWin)
  
  log_df <- x_df %>% inner_join(y_df,by=c("HTeamID"="ATeamID",
                                          "ATeamID"="HTeamID"))
  # note maybe come back and use glmnet? 
  # pythons uses l2 penalty
  log_reg <- glm(HWin ~ HDiff,data=log_df)
  coefs <- coef(log_reg)
  h <- (-coefs[1] / coefs[2]) / 2
  names(h) <- NULL
  return(list(a=coefs[2],b=coefs[1],h=h))
}

rescore_hfa <- function(df){
  hfa_mod = calc_hfa(df)
  hfa_value = hfa_mod$h/2
  game_dataframe <- df %>% mutate(
    HScore = HScore - hfa_value * NeutralFlag,
    AScore = AScore + hfa_value * NeutralFlag
  )
  return(game_dataframe)
}

rescore_ot <- function(df){
  df <- df %>% mutate(
    HScore = (HScore * (40 + NumOT * 5)) / 40,
    AScore = (AScore * (40 + NumOT * 5)) / 40
  )
  return(df)
}

### SRS - run it by season 
calc_srs <- function(df,rescore_ot=F,rescore_hfa=F){

  if(rescore_hfa){
    df <- rescore_hfa(df)
  }
  
  if(rescore_ot){
    df <- rescore_ot(df)
  }
  
  df2 <- df %>% mutate(
    HSpread = HScore - AScore,
    ASpread = -1 * HSpread
  )
  
  home_team <- df2 %>% select(HTeamID,HSpread,ATeamID) 
  colnames(home_team) <- c("TeamID","Spread","OppID")
  away_team <- df2 %>% select(ATeamID,ASpread,HTeamID)
  colnames(away_team) <- c("TeamID","Spread","OppID")
  
  combine <- bind_rows(home_team,away_team)
  average_spread <- combine %>% group_by(TeamID) %>% summarize(
    spread = mean(Spread)
  )
  all_teams <- average_spread %>% pull(TeamID)
  
  terms = matrix(ncol = length(all_teams))
  solutions = c()
  
  for(team in all_teams){
    row = c()
    opp_list <- combine %>% filter(TeamID==team) %>% select(OppID) %>% pull() 
    
    for(opp in all_teams){
      if(opp == team){
        row = c(row,1)
      }
      if(opp %in% opp_list){
        row = c(row,-1/length(opp_list))
      }
      else{
        row = c(row,0)
      }
    }
    team_spread <- average_spread[average_spread$TeamID==team,]$spread
    terms <- rbind(terms,row)
    solutions <- rbind(solutions,team_spread)
  }
  terms <- terms[-1,]
  rownames(terms) <- all_teams
  
  results_ex <- lsei(terms, solutions)
  
  srs_df <- data_frame(TeamID = rownames(terms),
                       SRS = results_ex)
  return(srs_df)
}


## RPI - Per Season 
calc_rpi <- function(df,rescore_ot=F){
  if(rescore_ot){
    df = rescore_ot(df)
  }
  
  df <- df %>% mutate(
    HWin = (HScore > AScore) * 1,
    AWin = 1 - HWin,
    HLoss = 1- HWin,
    ALoss = HWin
  )
  
  home_df <- df %>% select(HTeamID,ATeamID,HWin,HLoss)
  colnames(home_df) <- c("TeamID","OppID","HWin","HLoss")
  away_df <- df %>% select(ATeamID,HTeamID,AWin,ALoss) 
  colnames(away_df) <- c("TeamID","OppID","AWin","ALoss")
  teams_df <- bind_rows(home_df,away_df) 
  rpi <- teams_df %>% group_by(TeamID) %>%  summarise(across(HWin:ALoss, ~mean(.,na.rm=T))) %>% 
    mutate(
      W = HWin + AWin,
      L = HLoss + ALoss,
      WP = W/(W+L),
      AdjWP = (HWin * 0.6 + AWin * 1.4) / (HWin * 0.6 + AWin * 1.4 + HLoss * 1.4 + ALoss * 0.6)
    )
  
  rpi2 <- rpi %>% mutate(
    OWP = purrr::map_dbl(TeamID,~calc_owp(teams_df,rpi,.,F))
  )
  
  rpi3 <- rpi2 %>% mutate(
    OOWP = purrr::map_dbl(TeamID,~calc_owp(teams_df,rpi2,.,T)),
    RPI = 0.25 * AdjWP + 0.5 * OWP + 0.25 * OOWP
  ) 
  
  final_rpi <- rpi3 %>% select(TeamID,RPI) %>% mutate(TeamID=as.character(TeamID))
  return(final_rpi)
}

calc_owp <- function(teams,rpi,team_id,oowp=F){
  opp_team_list = teams %>% filter(TeamID==team_id) %>% pull(OppID)
  if(!oowp){
    owp <- rpi %>% filter(TeamID %in% opp_team_list) %>% pull(WP) %>% mean(.,na.rm=T)
    return(owp)
  }
  if(oowp){
    oowp <- rpi %>% filter(TeamID %in% opp_team_list) %>% pull(OWP) %>% mean(.,na.rm=T)
    return(oowp)
  }
  
}

calc_mov <- function(df,rescore_ot=F,rescore_hfa=F){
  if(rescore_hfa){
    df <- rescore_hfa(df)
  }
  
  if(rescore_ot){
    df <- rescore_ot(df)
  }
  
  df <- df %>% mutate(
    HSpread = HScore - AScore,
    ASpread = -1 * HSpread
  )
  
  home_team <- df %>% select(HTeamID,HSpread,ATeamID) 
  colnames(home_team) <- c("TeamID","Spread","OppID")
  away_team <- df %>% select(ATeamID,ASpread,HTeamID)
  colnames(away_team) <- c("TeamID","Spread","OppID")
  
  combine <- bind_rows(home_team,away_team)
  average_spread <- combine %>% group_by(TeamID) %>% summarize(
    MOV = mean(Spread)
  ) 
  return(average_spread)
}

calc_team_ratings <- function(df,szn,rescore_ot=F,rescore_hfa=F){
  if(rescore_hfa){
    df <- rescore_hfa(df)
  }
  
  if(rescore_ot){
    df <- rescore_ot(df)
  }
  year_conf <- team_conf %>% filter(Season==szn)
  
  df <- df %>% left_join(year_conf,by=c("ATeamID"="TeamID")) %>% 
    left_join(year_conf,by=c("HTeamID"="TeamID"),suffix=c("_away","_home")) 
  
  home_team <- df %>% select(HTeamID,ATeamID,HScore,NeutralFlag,DayNum,ConfAbbrev_home,ConfAbbrev_away) 
  colnames(home_team) <- c("Offense","Defense","Score","NeutralFlag","DayNum","Offense_Conf","Defense_Conf")
  home_team$loc <- "H"
  home_team$loc[home_team$NeutralFlag] <- "N"
  
  away_team <- df %>% select(ATeamID,HTeamID,AScore,NeutralFlag,DayNum,ConfAbbrev_away,ConfAbbrev_home) 
  colnames(away_team) <- c("Offense","Defense","Score","NeutralFlag","DayNum","Offense_Conf","Defense_Conf")
  away_team$loc <- "A"
  away_team$loc[away_team$NeutralFlag] <- "N"
  
  combine <- bind_rows(home_team,away_team) %>% 
    mutate(
      weight = exp(DayNum/20),
      in_conf_game = (Offense_Conf == Defense_Conf) * 1,
      int_off_conf_game = interaction(Offense_Conf,in_conf_game),
      int_def_conf_game = interaction(Defense_Conf,in_conf_game)
    )
  
  model <- lme4::lmer(Score ~ 0 + loc + (1|int_off_conf_game) + (1|int_def_conf_game) + (1|Offense) + (1|Defense),
                      data = combine,weights = weight)
  
  offense_ratings = lme4::ranef(model)$Offense
  offense_ratings$Team = row.names(lme4::ranef(model)$Offense)
  colnames(offense_ratings) = c('Off','TeamID')
  defense_ratings = lme4::ranef(model)$Defense
  defense_ratings$Team = row.names(lme4::ranef(model)$Defense)
  colnames(defense_ratings) = c('Def','TeamID')
  rtg_df = merge(offense_ratings, defense_ratings,on='TeamID')
  return(rtg_df)
}


create_all_rtgs <- function(season_df,season){
  print(season)
  mod = calc_hfa(season_df)
  # rescored so don't worry about it now
  season_df = rescore_hfa(season_df)
  season_df = rescore_ot(season_df)
  
  # calculate ratings now 
  srs = calc_srs(season_df) %>% mutate(TeamID = as.character(TeamID))
  rpi = calc_rpi(season_df) %>% mutate(TeamID = as.character(TeamID))
  mov = calc_mov(season_df) %>% mutate(TeamID = as.character(TeamID))
  team_rts = calc_team_ratings(season_df,season) %>% mutate(TeamID = as.character(TeamID))
  
  ts = calc_ts(season_df) %>% tibble::rownames_to_column(var="TeamID") %>% mutate(TeamID=as.character(TeamID))
  colley = calc_colley(season_df) %>% tibble::rownames_to_column(var="TeamID") %>% mutate(TeamID=as.character(TeamID))
  lrmc = calc_lrmc(season_df,mod) %>% tibble::rownames_to_column(var="TeamID") %>% mutate(TeamID=as.character(TeamID))
  elo = calc_elo(season_df) %>% tibble::rownames_to_column(var="TeamID") %>% mutate(TeamID=as.character(TeamID))
  
  all_metrics = list(srs,rpi,mov,team_rts,ts,colley,lrmc,elo)
  
  final_metrics = all_metrics %>% reduce(left_join, by = "TeamID") 
  return(final_metrics)
}


df <- read_csv("2022/kaggle_data/MRegularSeasonCompactResults.csv") %>% 
  group_by(Season) %>% nest() %>% 
  mutate(
    data = purrr::map(data,refactor_game_dataframe),
    final_rts = purrr::map2(data,Season,create_all_rtgs)
  ) %>% select(-data)
  
saveRDS(df,"2022/full_season_basic_rtgs.RDS")

