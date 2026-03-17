cbb_temp <- read_csv("../march_madness/2024/data/ncaabb23.csv") 

cbb_temp <- cbb_temp |> 
  mutate(across(c("home","road"), ~gsub("[.]","",toupper(.)))) |> 
  mutate(home_win = ifelse(hscore>rscore,1,0),
         date = as.Date(date, format = "%m/%d/%Y"),
         day = as.numeric(difftime(date, min(date), units = "days"))+1,
         weight = (day^4)/max(day^4))

unique_teams <- sort(union(cbb_temp$home, cbb_temp$road))

game_matrix <- matrix(0, nrow=dim(cbb_temp)[1], ncol = length(unique_teams))


# this is vectorized so easy, no need to loop
home_indices <- match(cbb_temp$home, unique_teams)
away_indices <- match(cbb_temp$road, unique_teams)

# Use matrix indexing to assign 1 for home team and -1 for away team
game_matrix[cbind(1:nrow(cbb_temp), home_indices)] <- 1
game_matrix[cbind(1:nrow(cbb_temp), away_indices)] <- -1


# interesting didn't know you could add a matrix to the regression
fit_spread <- lm(line ~ 1 + game_matrix + neutral, data = cbb_temp, weights = cbb_temp$weight)

# interesting didn't know you could add a matrix to the regression
fit_spread <- lm(line ~ 1 + game_matrix + neutral, data = cbb_temp, weights = cbb_temp$weight)

betas <- fit_spread$coefficients
betas[length(betas)-1] <- 0 

names(betas) <- c("int",unique_teams, "Neutral")

# could go broom route, but harder to map to team names maybe
pred_spread2 <- betas |> tibble::enframe() |> 
  mutate(rank = dense_rank(desc(value)))



fit_win_prob <- glm(home_win ~ 0 + line, data = cbb_temp, family = "binomial")

#fit_gam_win_prob <- mgcv::gam(home_win ~ 0 + s(line), data = cbb_temp, family = "binomial")

m_seeds <- read_csv("../march_madness/2024/data/2024_tourney_seeds.csv") |> filter(Tournament=="M")


# It's Wagner not howard
m_seeds[m_seeds$Seed=="X16","TeamID"] <- 1447

m_spelling <- read_csv("../march_madness/2024/data/MTeamSpellings.csv") |> rename(name="TeamNameSpelling") |> 
  mutate(name = iconv(name, "UTF-8", "ASCII", sub = ""),
         name = gsub("[.]","",name))

reg_coefs <- pred_spread2 |> filter(name %in% c("int","Neutral"))
team_coefs <- pred_spread2 |> anti_join(reg_coefs) |> 
  mutate(name = tolower(name)) |> 
  left_join(m_spelling) |> 
  distinct(TeamID, .keep_all = TRUE)

pred_prob_fn <- function(x, wp_model) {
  out <- exp(coef(wp_model) * x) / (1 + exp(coef(wp_model) * x))
  return(out)
}

potential_bracket <- m_seeds |>
  mutate(join = 1) |>
  inner_join(
    m_seeds |>
      mutate(join = 1),
    by = join_by("join"),
    suffix = c("_a", "_b"),
    relationship = "many-to-many"
  ) |>
  filter(TeamID_a != TeamID_b) |>
  filter(TeamID_a < TeamID_b) |>
  mutate(ID = paste0(2024, "_", TeamID_a, "_", TeamID_b)) |> 
  left_join(team_coefs |> select(TeamID, value),by=c("TeamID_a"="TeamID")) |> 
  left_join(team_coefs |> select(TeamID, value),by=c("TeamID_b"="TeamID"),suffix=c("_a","_b")) |> 
  mutate(team_spread = value_a - value_b,
         total_spread = team_spread + sum(reg_coefs$value),
         pred_prob = purrr::map_dbl(total_spread, function(i){
           pred_prob_fn(i, wp_model = fit_win_prob)
         })) 

write_csv(potential_bracket,"../march_madness/2024/bracket_predictions.csv")


## Clean up seeds and implied probs

df_impl_probs <- read_csv("../march_madness/2024/implied_probs.csv")
m_teams <- read_csv("../march_madness/2024/data/MTeams.csv")

df_final <- df_impl_probs |> left_join(m_seeds,by=c("Team"="Seed")) |> 
  left_join(m_teams |> select(TeamID,TeamName)) |> 
  left_join(team_coefs |> select(TeamID, value),by=c("TeamID")) |> 
  mutate(
    bracket_region = substr(Team,1,1),
    seed = as.numeric(substr(Team,2,3))
  )

colnames(df_final)

library(gt)
library(cbbplotR)

## include points above average
create_gt_table <- function(df_tbl, region_name){
  regions_mapping <- c("W"="East",
                       "Y"="Midwest",
                       "X"="West",
                       "Z"="South")
  
  region_tbl_name <- regions_mapping[region_name]
  
  df_tbl |> select(-bracket_region) |> 
    arrange(desc(R1)) |> 
    gt_cbb_teams("TeamName", "TeamName") |> 
    gt() |> 
    fmt_percent(columns = starts_with("R")) |> 
    fmt_number(columns = "value",decimals = 2) |> 
    fmt_markdown(TeamName) |> 
    data_color(
      columns = c(value),
      fn = scales::col_numeric(
        palette = 'PuBuGn',
        domain = c(min(df_tbl$value), max(df_tbl$value)),
      )
    ) |> 
    data_color(
      columns = starts_with("R"),
      fn = scales::col_numeric(
        palette = ggsci::rgb_material('amber', n = 68),
        domain = c(0,1),
      )
    ) |> 
    gtExtras::gt_theme_nytimes() |> 
    tab_source_note("Based on 1000 Simulations of NCAA Tournament") |> 
    tab_header(
      title = md("**2024 NCAA Men's Basketball Tournament Odds**"),
      subtitle = md(glue::glue("{region_tbl_name}"))
    ) |> 
    cols_align(align = "left") |> 
    cols_label(
      TeamName = "",
      seed = "Seed",
      value = "Points Above Avg",
      R1 = 'R64',
      R2 = 'R32',
      R3 = 'Sweet 16',
      R4 = 'Elite 8',
      R5 = 'Final 4',
      R6 = 'NCG'
    )
}

lst_bracket <- df_final |>  select(TeamName,seed,value,bracket_region,starts_with("R"))  |> group_split(bracket_region)

create_gt_table(lst_bracket[[1]],"W") |> gtsave("east_region.png")
create_gt_table(lst_bracket[[2]],"X") |> gtsave("west_region.png")
create_gt_table(lst_bracket[[3]],"Y") |> gtsave("midwest_region.png")
create_gt_table(lst_bracket[[4]],"Z") |> gtsave("south_region.png")
