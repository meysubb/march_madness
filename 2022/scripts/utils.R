refactor_game_dataframe <- function(df){
  location_split <- split(df,df$WLoc)
  
  location_cleanup <- lapply(seq_along(location_split),function(i){
    df <- location_split[[i]]
    name <- names(location_split[i])
    df$NeutralFlag <- FALSE
    if(name=="N"){
      df$NeutralFlag <- TRUE
    }
    # for neutral and home games
    colnames(df) <- c('DayNum','HTeamID','HScore','ATeamID','AScore','WLoc','NumOT','NeutralFlag')
    if(name=="A"){
      colnames(df) <- c('DayNum','ATeamID','AScore','HTeamID','HScore','WLoc','NumOT','NeutralFlag')
      df$loc <- "A"
    }
    if(name=="H"){
      df$loc <- "H"
    }
    if(name=="N"){
      df$loc <- "N"
    }
    return(df)
  })
  full_df <- bind_rows(location_cleanup) %>% select(-WLoc) %>% arrange(-DayNum)
  return(full_df)
}


refactor_game_detailed <- function(df){
  location_split <- split(df,df$WLoc)
  
  location_cleanup <- lapply(seq_along(location_split), function(i){
    df <- location_split[[i]]
    name <- names(location_split[i])
    df$NeutralFlag <- FALSE
    if(name=="N"){
      df$NeutralFlag <- TRUE
      colnames(df) <-
        c(
          'DayNum','HTeamID','HScore','ATeamID','AScore','WLoc','NumOT',
          'HFGM','HFGA','HFGM3','HFGA3','HFTM','HFTA','HOR','HDR','HAST','HTO','HSTL','HBLK','HPF',
          'AFGM','AFGA','AFGM3','AFGA3','AFTM','AFTA','AOR','ADR','AAST','ATO','ASTL','ABLK','APF',
          'NeutralFlag'
        )
    }
    if(name == "H"){
      colnames(df) <- c(
        'DayNum','HTeamID','HScore','ATeamID','AScore','WLoc','NumOT',
        'HFGM','HFGA','HFGM3','HFGA3','HFTM','HFTA','HOR','HDR','HAST','HTO','HSTL','HBLK','HPF',
        'AFGM','AFGA','AFGM3','AFGA3','AFTM','AFTA','AOR','ADR','AAST','ATO','ASTL','ABLK','APF',
        'NeutralFlag')
    }
    if(name == "A"){
      colnames(df) <- c(
        'DayNum','HTeamID','HScore','ATeamID','AScore','WLoc','NumOT',
         'AFGM','AFGA','AFGM3','AFGA3','AFTM','AFTA','AOR','ADR','AAST','ATO','ASTL','ABLK','APF',
         'HFGM','HFGA','HFGM3','HFGA3','HFTM','HFTA','HOR','HDR','HAST','HTO','HSTL','HBLK','HPF',
         'NeutralFlag'
      )
    }
    return(df)
  })
  full_df <- bind_rows(location_cleanup) %>% select(-WLoc) %>% arrange(-DayNum)
  return(full_df)
}
