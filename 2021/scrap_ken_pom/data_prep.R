pacman::p_load(googlesheets, dplyr, stringr, readr, tidyr, janitor, purrr,stringr)

ken_pom_data_sheet <- gs_key("1AauzEVB-T01TqI2hY81sT6i-gloLwPbTnh8tqsw-TYY")

# latest data is on first sheet, older years on subsequent sheets
get_kp_sheet <- function(sheet_n) {
  ken_pom_data_sheet %>%
    gs_read_cellfeed(ws = sheet_n) %>% 
    { # grab title of worksheet and include it as a column
      ws <- attr(., "ws_title")
      gs_reshape_cellfeed(.) %>% 
        mutate(source_dat = ws)      
    }
  
}    

year_seq <- seq_along(2012:2019)

# this will grab all years of data from Google Sheets - takes time
kp_all_years_raw <- year_seq %>% 
  map(get_kp_sheet)

# from google sheets 2014-2019
data=kp_all_years_raw %>% bind_rows()
colnames(data) = data[1,]
colnames(data)[length(data)] = "Year"
colnames(data)[c(14,16,18,20)] = c('SOS_AdjEM',"SOS_OppO","SOS_OppD","NCSOS_AdjEM")
data=data %>% select(
  colnames(data)[colnames(data) != 'NA'])
data=data %>% filter(
  Rk != 'Rk'
) %>% mutate(
  Year = gsub("_Final", "",Year),
)
lst = c("Rk","AdjEM","AdjO","AdjD","AdjT","Luck","SOS_AdjEM","SOS_OppO","SOS_OppD","NCSOS_AdjEM","Year")
data[lst] <- data[lst] %>% 
  mutate_if(is.character,as.numeric)
#  from csv 2020-2021

setwd("~/GitHub/march-madness-2021/scrap_ken_pom")

temp = list.files(path="raw/",pattern="*.csv")

d1=read.csv(paste0("raw/",temp[1]));d2=read.csv(paste0("raw/",temp[2]));data_new =rbind(d1,d2)

colnames(data_new)[c(14,16,18,20)] = c('SOS_AdjEM',"SOS_OppO","SOS_OppD","NCSOS_AdjEM")

data_new=data_new %>% select(
  colnames(data_new)[str_detect(colnames(data_new),'NA')==FALSE]
)

colnames(data_new)[1] = 'Rk'

colnames(data_new)[4] = 'W-L'

lst = c("Rk","AdjEM","AdjO","AdjD","AdjT","Luck","SOS_AdjEM","SOS_OppO","SOS_OppD","NCSOS_AdjEM","Year")
data_new[lst] <- data_new[lst] %>% 
  mutate_if(is.character,as.numeric)

all_data = rbind(data,data_new)
all_data=all_data %>% arrange(Year,Rk)

all_data=all_data %>% 
  mutate(
    Team = str_remove(Team,"Â"),
    Team = str_remove(Team,"[0-9]"),
    Team = str_remove(Team,"[0-9]"),
    Team = str_remove(Team,"[*]"),
    Team = str_remove(Team,"[*]"),
    Team = str_trim(Team)
  )


write.csv(all_data,"ken_pom_preID.csv",row.names = F)

stringr::str_remove('ads* dw * deD *','[*]')

all_data %>% head(10)
all_data %>% tail(10)
