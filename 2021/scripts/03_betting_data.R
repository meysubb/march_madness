library(tidyverse)

betting_files <-
  list.files(path = "betting_data/prediction_tracker/")

empty_data_frame <- data.frame()
for(i in seq_along(betting_files)){
  csv_file <- glue::glue("betting_data/prediction_tracker/{betting_files[i]}")
  numeric_year <- as.numeric(paste0("20",str_extract(csv_file,"[0-9]+")))
  
  data <- read_csv(csv_file) %>%
    select(date, home, hscore, road, rscore, neutral, lineopen) %>%
    readr::type_convert() %>%
    mutate(
      date = as.Date(date, format = "%m/%d/%Y"),
      year = lubridate::year(date),
      score_diff = hscore - rscore,
      spread_sign = sign(lineopen),
      score_diff_sign = sign(score_diff),
      h_win_against_spread = ifelse(score_diff > lineopen, T, F),
      a_win_against_spread = !h_win_against_spread
    )
  
  
  home_team <- data %>% select(date, home, h_win_against_spread) 
  away_team <- data %>% select(date, road, a_win_against_spread)
  
  colnames(home_team) <- c("date","team","win_against_spread")
  colnames(away_team) <- c("date","team","win_against_spread")
  
  team_df <- bind_rows(home_team,away_team) %>% 
    group_by(team) %>% 
    summarize(
      year = numeric_year,
      total_games = n(),
      total_wins_spread = sum(win_against_spread,na.rm = T),
      pct_win_spread = total_wins_spread/total_games
    )
  
  empty_data_frame <- bind_rows(empty_data_frame,team_df)
  
}

team_spellings <- read_csv("kaggle_data/MTeamSpellings.csv")

betting_ats_data <- empty_data_frame %>% 
  mutate(team = tolower(team)) %>% left_join(team_spellings,
                                             by = c("team"="TeamNameSpelling"))

write_csv(betting_ats_data,"betting_data/win_ats_teams_historical.csv")

