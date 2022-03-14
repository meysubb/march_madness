library(tidyverse)
season_df <- read_csv("kaggle_data/MRegularSeasonDetailedResults.csv") 

season_df <- season_df %>% filter(Season >=2010) %>% mutate(LLoc = case_when("WLoc"=="H" ~ "A",
                                                                             "LLoc" == "A" ~ "H",
                                                                             TRUE ~ "N"))

source("scripts/01_preprocessing.R")

create_factors <- function(df){
  df <- df %>% mutate(
    team_rf_id = ifelse(
      team_rf %in% create_residual_factors(df,"team_rf",20),
      team_rf,
      "999999"
    )
  )
  return(df)
}

create_residual_factors <- function(df,col,min_games){
  cnt <- table(df[,col])
  cnt_names <- names(cnt)
  res_teams <- cnt_names[cnt >=  min_games]
}


season_df <- create_advanced_metrics(season_df) %>% mutate(w_score_diff = w_score - l_score,
                                                           l_score_diff = l_score - w_score)

w_games <- season_df %>% select(season,day_num,starts_with("w")) %>% mutate(w_win = 1)
l_games <- season_df %>% select(season,day_num,starts_with("l")) %>% mutate(l_win = 0)

colnames(w_games) <- gsub("^w","",colnames(w_games))
colnames(l_games) <- gsub("^l","",colnames(l_games))

games <- bind_rows(w_games,l_games)
colnames(games) <- gsub("^_","",colnames(games))


## run a simple random effects model to see how much better specific teams are
colnames(games)

train_df <- games %>% select(season,day_num,team_id,off_rtg:win) %>%
  mutate(team_rf = paste0(season,"_",team_id))

train_df2 <- create_factors(train_df) %>% mutate(
  team_rf_id = as.factor(team_rf_id),
  sign = sign(score_diff),
  score_diff = if_else(abs(score_diff)>=20,sign * 20,score_diff),
  
) 



library(brms)

# model <- brm(score_diff ~ off_rtg + def_rtg + ast_ratio + tor_ratio + efg + ftrate + orbp + drp + rebp + four_factor + 
#                (1 | team_rf_id),
#              data = train_df2,
#              family = gaussian(),
#              sample_prior = T,
#              chains = 4,
#              cores = 4,
#              backend="cmdstanr")

library(lme4)
model_glme <- glmer(win ~ off_rtg + def_rtg + ast_ratio + tor_ratio + efg + ftrate + orbp + drp + rebp + four_factor +
                    (1 | team_rf_id),
                  data = train_df2,
                  family = binomial)

model_lme <-
  lmer(
    score_diff ~ off_rtg + def_rtg + ast_ratio + tor_ratio + efg + ftrate + orbp + drp + rebp + four_factor +
      (1 | team_rf_id),
    data = train_df2)
# 
# summary(model_lme)
# 
# ran_effs = ranef(model_lme)[[1]]
# 
# ran_effs_clean <-
#   ran_effs %>% tibble::rownames_to_column(var = "team_season_id") %>% tidyr::separate(team_season_id,
#                                                                                           c("team_id", "season"), "_")

## Try BPCS tomorrow