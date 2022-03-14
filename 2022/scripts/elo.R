library(elo)
library(dplyr)
library(readr)

source("2022/scripts/utils.R")

tourney_raw <- read_csv("2022/kaggle_data/MNCAATourneyCompactResults.csv")
season_raw <- read_csv("2022/kaggle_data/MRegularSeasonCompactResults.csv")

season_df <-
  refactor_game_dataframe(season_raw)

tourney_df <- refactor_game_dataframe(tourney_raw) 


complete_df <- bind_rows(season_raw,tourney_raw) %>%  arrange(-Season,-DayNum)


# elo <- elo.run(score(WScore, LScore) ~ 
#                  as.character(WTeamID) + 
#                  as.character(LTeamID) +
#                  k(20*((WScore - LScore + 3)^.8)/
#                      (7.5+.006*
#                         case_when(WLoc == "H" ~ 100,
#                                   WLoc == "A" ~ -100,
#                                   TRUE ~ 0))
#                  ) + 
#                  regress(Season, 1500, 0) + 
#                group(Season),
#                data = complete_df)
# 
# elo_data <- as.data.frame(elo) %>% select(team.A,team.B,elo.A,elo.B)
# 
# final_elo <- complete_df %>% bind_cols(elo_data)




## Offensive/Defensive Ratings (Efficiency adjusted)?
df <- bind_rows(season_df,tourney_df)

home_df <- df %>% select(HTeamID,ATeamID,HScore,NeutralFlag) %>% 
  mutate(
    NeutralFlag = if_else(NeutralFlag,"N","H")
  )
colnames(home_df) <- c("Offense","Defense","Score","Neutral")

away_df <- df %>% select(ATeamID,HTeamID,AScore,NeutralFlag) %>% 
  mutate(
    NeutralFlag = if_else(NeutralFlag,"N","A")
  )
colnames(away_df) <- c("Offense","Defense","Score","Neutral")

scoring = bind_rows(home_df,away_df)

## maybe run this by season (not including tournament information)?
model <- lme4::lmer(Score ~ 0 + Neutral + (1|Offense) + (1|Defense),
                    data = scoring)
