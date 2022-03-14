
pacman::p_load(dplyr,stringr,tidyr)

setwd("~/GitHub/march-madness-2021")

files=list.files(path="./modeling_ncaa_ml/data")

results=read.csv(paste0("modeling_ncaa_ml/data/",files[1])) %>% 
  filter(
    Season >= 2012,
    DayNum %in% c(136,137)
    ) %>%
  mutate(
    big_id = if_else(WTeamID > LTeamID,WTeamID,LTeamID),
    small_id = if_else(WTeamID < LTeamID,WTeamID,LTeamID),
    smallid_won = if_else(small_id == WTeamID,1,0),
    matchbigTeam = paste0(Season,big_id),
    matchsmallTeam = paste0(Season,small_id)
  ) %>% select(-WTeamID,-WScore,-LTeamID,-LScore,-WLoc,-NumOT)


pom=read.csv("scrap_ken_pom/ken_pom_ready.csv")
pom=pom %>%
  mutate(
    Team = stringr::str_replace(Team,"[.]",""),
    Team = stringr::str_replace(Team,"[.]",""),
    Team = stringr::str_replace(Team,"Saint","St")
  )

ids = read.csv(paste0("modeling_ncaa_ml/data/",files[2]))%>% rename(Team = TeamName)

#drop some variables because I didn't wan5 to deal with diff names yet
pom=dplyr::left_join(pom,ids, by = "Team") %>% drop_na()%>%
  mutate(
    match = paste0(Year,TeamID),
  )
results = left_join(results,pom,by = c('matchbigTeam'='match'))
results = left_join(results,pom,by = c('matchsmallTeam'='match'),suffix = c('_bigid','_smallid'))
results = results %>% drop_na()
results=results %>% select(
  -FirstD1Season_bigid,-LastD1Season_bigid,-Year_smallid,-Year_bigid,
  -LastD1Season_smallid,-FirstD1Season_smallid,-matchbigTeam,-matchsmallTeam,
  -TeamID_bigid,-TeamID_smallid)

write.csv(results,'modeling_ncaa_ml/data_for_modeling.csv',row.names = F)






