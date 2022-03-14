library(tidyverse)
library(janitor)

# season_df <- read_csv("kaggle_data/MRegularSeasonDetailedResults.csv") 
# tourney_df <- read_csv("kaggle_data/MNCAATourneyCompactResults.csv")
# seed_df <- read_csv("kaggle_data/MNCAATourneySeeds.csv")
# ken_pom <- read_csv("scrap_ken_pom/ken_pom_ready.csv")
# 
# 
# test = season_df %>% dplyr::filter(Season == 2020)
# 


# this function should create all advanced metrics for any game 
# regular season and tourney 
create_advanced_metrics <- function(detailed_games){
  detailed_games <- detailed_games %>% clean_names() %>%
    mutate_at(vars(contains('id')), funs(as.factor))
  
  final_metrics <- detailed_games %>% mutate(
    # possessions
    w_poss = 0.96*(wfga + wto + 0.44*wfta - wor),
    l_poss = 0.96*(lfga + lto + 0.44*lfta - lor),
    # Offensive efficiency (OffRtg) = 100 x (Points / possessions)
    # Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
    # Net Rating = Off.Rtg - Def.Rtg
    w_off_rtg = 100 * (w_score / w_poss),
    l_off_rtg = 100 * (l_score / l_poss),
    w_def_rtg = l_off_rtg, 
    l_def_rtg = w_off_rtg,
    # we could create rolling 5 game netrtgs for teams 
    w_net_rtg = w_off_rtg - w_def_rtg,
    l_net_rtg = l_off_rtg - l_def_rtg,
    # assist ratio = Percentage of team possessions that end in assists
    w_ast_ratio = 100 * w_ast / (wfga + 0.44*wfta + w_ast + wto),
    l_ast_ratio = 100 * l_ast / (lfga + 0.44*lfta + l_ast + lto),
    # turnover ratio = (TO * 100) / (FGA + (FTA * 0.44) + AST + TO) (to per 100 poss)
    w_tor_ratio = 100 * wto / (wfga + 0.44*wfta + w_ast + wto),
    l_tor_ratio = 100 * lto / (lfga + 0.44*lfta + l_ast + lto),
    #eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
    w_efg = (wfgm + 0.5 * wfgm3) / wfga,
    l_efg = (lfgm + 0.5 * lfgm3) / lfga,
    #FTA Rate : How good a team is at drawing fouls.
    w_ftrate = wfta / wfga,
    l_ftrate = lfta / lfga,
    #OREB% : Percentage of team offensive rebounds
    w_orbp = wor / (wor + ldr),
    l_orbp = lor / (lor + wdr),
    #DREB% : Percentage of team defensive rebounds
    w_drp = wdr / (wdr + lor), 
    l_drp = ldr / (ldr + wor),
    #REB% : Percentage of team total rebounds
    w_rebp = (wdr + wor) / (wdr + wor + ldr + lor),
    l_rebp = (ldr + lor) / (wdr + wor + ldr + lor),
    # 4factor
    w_four_factor = .40 * w_efg + .25 * w_tor_ratio + .15 * w_ftrate,
    l_four_factor = .40 * l_efg + .25 * l_tor_ratio + .15 * l_ftrate,
  )
}


# test_adv <- create_advanced_metrics(test)

## helper function to join data back
## df_b is always the one that is joined twice
## i.e df_a is the main dataframe
join_tables_on_id <- function(dataframe_a,dataframe_b,x_replacement="w_",y_replacement="l_"){
  
  first_team_id <- paste0(x_replacement,"_team_id")
  second_team_id <- paste0(y_replacement,"_team_id")
  join_cols <- "team_id"
  join_cols2 <- "team_id"
  names(join_cols) <- first_team_id
  names(join_cols2) <- second_team_id
  
  df_joined = dataframe_a %>%
    left_join(dataframe_b, by = c(join_cols, "season")) %>%
    left_join(dataframe_b, by = c(join_cols2, "season"))
  
  # rename columns
  w_ids <- str_detect(colnames(df_joined),".x")
  l_ids <- str_detect(colnames(df_joined),".y")
  
  colnames(df_joined)[w_ids] <- paste0(x_replacement,str_replace_all(colnames(df_joined)[w_ids],".x",""))
  colnames(df_joined)[l_ids] <- paste0(y_replacement,str_replace_all(colnames(df_joined)[l_ids],".y",""))
  
  return(df_joined)
}


# this function will join seed information to the compact tourney results
join_seed_information <- function(tourney_compact,
                                  tourney_seeds,
                                  x_name,
                                  y_name) {
  tourney_compact <- tourney_compact %>% janitor::clean_names()
  
  tourney_seeds = tourney_seeds %>% janitor::clean_names() %>%
    mutate(seed_numerical = as.factor(str_extract(seed, "[0-9]+")),
           seed_region = as.factor(str_extract(seed, "[a-zA-Z]"))) %>% select(-seed)
  
  tourney_details <-
    join_tables_on_id(
      tourney_compact,
      tourney_seeds,
      x_replacement = x_name,
      y_replacement = y_name
    )
  return(tourney_details)
}


# tourney_full <- join_seed_information(tourney_df, seed_df,x_name="w",y_name="l")

# this function will join KenPom data to whatever you want
join_ken_pom_information <-
  function(tourney_data,
           ken_pom_data,
           ken_pom_latest_year,
           x_name,
           y_name) {
    tourney_data <-
      tourney_data %>% janitor::clean_names() %>% filter(season >= ken_pom_latest_year)
    ken_pom_data <-
      ken_pom_data %>% janitor::clean_names() %>% rename(season = year)
    
    tourney_kp <-
      join_tables_on_id(tourney_data,
                        ken_pom_data,
                        x_replacement = x_name,
                        y_replacement = y_name)
    return(tourney_kp)
  }



# test = join_ken_pom_information(tourney_2012,ken_pom,2012)
