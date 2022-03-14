source("2022/scripts/utils.R")
library(tidyverse)
library(lme4)

team_conf <- read_csv("2022/kaggle_data/MTeamConferences.csv")


create_metrics <- function(game_df,season){
  year_conf = team_conf %>% filter(Season==season)
  
  game_df <- game_df %>% left_join(year_conf,by=c("ATeamID"="TeamID")) %>% 
    left_join(year_conf,by=c("HTeamID"="TeamID"),suffix=c("_away","_home")) 
  
  
  df2 <- game_df %>% mutate(
    HPOS = (HFGA - HOR) + HTO + (0.44 * HFTA),
    APOS = (AFGA - AOR) + ATO + (0.44 * AFTA),
    HFGM_POS = HFGM / HPOS,
    HFGA_POS = HFGA / HPOS,
    HFGM3_POS = HFGM3 / HPOS,
    HFGA3_POS = HFGA3 / HPOS,
    HFTM_POS = HFTM / HPOS,
    HFTA_POS = HFTA / HPOS,
    HOR_POS = HOR / HPOS,
    HDR_POS = HDR / APOS, # denominator is opponent possesions because these are a defensive stat
    HAST_POS = HAST / HPOS,
    HTO_POS = HTO / HPOS,
    HSTL_POS = HSTL / APOS, # likewise
    HBLK_POS = HBLK / APOS,
    HPF_POS = HPF / (HPOS + APOS), # denom is all possessions
    HPTS_POS = HScore / HPOS,
    AFGM_POS = AFGM / APOS,
    AFGA_POS = AFGA / APOS,
    AFGM3_POS = AFGM3 / APOS,
    AFGA3_POS = AFGA3 / APOS,
    AFTM_POS = AFTM / APOS,
    AFTA_POS = AFTA / APOS,
    AOR_POS = AOR / APOS,
    ADR_POS = ADR / HPOS,
    AAST_POS = AAST / APOS,
    ATO_POS = ATO / APOS,
    ASTL_POS = ASTL / HPOS,
    ABLK_POS = ABLK / HPOS,
    APF_POS = APF / (HPOS + APOS),
    APTS_POS = AScore / APOS,
    HTEMPO = 40 * HPOS / (40 + NumOT * 5),
    ATEMPO = 40 * APOS / (40 + NumOT * 5)
  )
  
  home_df <-
    df2 %>% select(
      TeamID = HTeamID,
      OppID = ATeamID,
      DayNum,
      OffenseConf = ConfAbbrev_home,
      DefenseConf = ConfAbbrev_away,
      FGM_POS = HFGM_POS,
      FGA_POS = HFGA_POS,
      FGM3_POS = HFGM3_POS,
      FGA3_POS = HFGA3_POS,
      FTM_POS = HFTM_POS,
      FTA_POS = HFTA_POS,
      OR_POS = HOR_POS,
      DR_POS = HDR_POS,
      AST_POS = HAST_POS,
      TO_POS = HTO_POS,
      STL_POS = HSTL_POS,
      BLK_POS = HBLK_POS,
      PF_POS = HPF_POS,
      PTS_POS = HPTS_POS,
      TEMPO = HTEMPO,
      NeutralFlag
    ) %>% 
    mutate(loc="H")
  home_df$loc[home_df$NeutralFlag] <- "N"
  
  
  away_df <-
    df2 %>% select(
      TeamID = ATeamID,
      OppID = HTeamID,
      DayNum,
      OffenseConf = ConfAbbrev_away,
      DefenseConf = ConfAbbrev_home,
      FGM_POS = AFGM_POS,
      FGA_POS = AFGA_POS,
      FGM3_POS = AFGM3_POS,
      FGA3_POS = AFGA3_POS,
      FTM_POS = AFTM_POS,
      FTA_POS = AFTA_POS,
      OR_POS = AOR_POS,
      DR_POS = ADR_POS,
      AST_POS = AAST_POS,
      TO_POS = ATO_POS,
      STL_POS = ASTL_POS,
      BLK_POS = ABLK_POS,
      PF_POS = APF_POS,
      PTS_POS = APTS_POS,
      TEMPO = ATEMPO,
      NeutralFlag
    ) %>% 
    mutate(loc="A")
 
  away_df$loc[away_df$NeutralFlag] <- "N"
    
  full_df <- bind_rows(home_df,away_df) %>% 
    mutate(
      weight = exp(DayNum/20),
      in_conf_game = (OffenseConf == DefenseConf) * 1,
      int_off_conf_game = interaction(OffenseConf,in_conf_game),
      int_def_conf_game = interaction(DefenseConf,in_conf_game)
    )
  
  terms = c('FGM_POS','FGA_POS','FGM3_POS','FGA3_POS','FTM_POS','FTA_POS','OR_POS','DR_POS','AST_POS','TO_POS','STL_POS',
           'BLK_POS','PF_POS','PTS_POS','TEMPO')
  
  set.seed(41)
  data <- full_df[,terms]
  scaled_data <- scale(data)
  
  clus_k <- kmeans(scaled_data,centers=4)
  full_df$cluster <- clus_k$cluster
  
  
  df_list = list()
  for (term in terms) {
    print(glue::glue("Create ratings for {term} in {season}"))
    term_rtg_df = create_stat_ratings(full_df,term_str = term)
    df_list[[length(df_list)+1]] <- term_rtg_df
  }
  final_df = df_list %>% reduce(left_join, by = "TeamID")   
  return(final_df)
}



create_stat_ratings <- function(df,term_str){
  
  formula = as.formula(paste0(term_str,' ~ 0 + loc + in_conf_game + (1|cluster) + (1|TeamID) + (1|OppID) '))
  model <- lme4::lmer(formula,
                      data = df,weights = weight)
  
  made_ratings = lme4::ranef(model)$TeamID
  made_ratings$Team = row.names(lme4::ranef(model)$TeamID)
  colnames(made_ratings) = c(paste0(term_str,'Made'),'TeamID')
  allowed_ratings = lme4::ranef(model)$OppID
  allowed_ratings$Team = row.names(lme4::ranef(model)$OppID)
  colnames(allowed_ratings) = c(paste0(term_str,'Allowed'),'TeamID')
  
  rtg_df = merge(made_ratings, allowed_ratings ,on='TeamID')
  return(rtg_df)
}


create_massey <- function(massey_df){
  massey_year_split = massey_df %>% slice_max(RankingDayNum) %>% select(SystemName,TeamID,OrdinalRank) %>% 
    pivot_wider(names_from = SystemName,
                values_from = OrdinalRank)
  return(massey_year_split)
}


df <-
  read_csv("2022/kaggle_data/MRegularSeasonDetailedResults.csv") %>%  group_by(Season) %>% nest() %>%
  mutate(
    data = purrr::map(data, refactor_game_detailed),
    final_rts = purrr::map2(data, Season, create_metrics)
  ) %>% select(-data) %>% unnest()

saveRDS(df,"2022/adv_rtgs.RDS")


massey_df <- read_csv("2022/kaggle_data/MMasseyOrdinals.csv") %>% group_by(Season) %>% nest() %>%  
  mutate(
    massey_ords = purrr::map(data,create_massey)
  ) %>% select(-data) %>% unnest()

saveRDS(massey_df,"2022/mass_rtgs.RDS")