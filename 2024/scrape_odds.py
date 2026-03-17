#import packages
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json as js
from datetime import datetime, timedelta
import time

start = datetime(2023, 12, 27)
end = datetime(2024, 3, 17)

dateInputs = [
    (start + timedelta(days=i)).strftime("%m-%d-%Y")
    for i in range((end - start).days + 1)
]

bookIds = {10 : 'Pinn',
        27 : 'BM',
        3 : 'BOL',
        13 : 'PPH',
        30 : 'Circa'}

marketIds = {1 : 'Spread',
            2 : 'Total',
            3 : 'ML'}

def scrape_odds_between(dateInputs, bookIds, marketIds):
    for dateInput in dateInputs:
        url = 'https://pregame.com/api/gamecenter/init?dt=' + dateInput
        print(url)
        r = requests.get(url)
        data = r.text
        soup = BeautifulSoup(data, 'lxml')
        jsData = js.loads(soup.text)

        events = [event for event in jsData['GameCenterData']['Events'] if event.get('LeagueId') == 4]
        events_df = pd.DataFrame(events)
        events_df = events_df.filter(items = ['Id', 'EventGroupId', 'LeagueId', 'LeagueName',
                                        'ScheduleDateAndTime', 'StartDateAndTime',
                                        'AwayRotationNumber', 'AwayTeamId', 'AwayTeamName',
                                        'AwayTeamFullName', 'AwayTeamAbbr', 'HomeRotationNumber',
                                        'HomeTeamId', 'HomeTeamName', 'HomeTeamFullName', 'HomeTeamAbbr',
                                        'AtNeutralSite', 'Status'])

        odds = [odd for odd in jsData['GameCenterData']['Odds'] if odd.get('LeagueId') == 4]
        odds_df = pd.DataFrame(odds)
        odds_df['BookName'] = odds_df['BookId'].map(bookIds)
        odds_df['MarketName'] = odds_df['ActionTypeId'].map(marketIds)

        odds_df = odds_df.filter(items = ['EventId', 'LeagueId', 'BookId', 
                                    'PeriodTypeId', 'CurrentAwayPoints',
                                    'CurrentAwayPrice', 'CurrentHomePoints',
                                    'CurrentHomePrice', 'OpenerAwayPoints',
                                    'OpenerAwayPrice', 'OpenerHomePoints',
                                    'OpenerHomePrice', 'BookName', 'MarketName'])

        odds_df = odds_df[odds_df['BookName'].isin(bookIds.values())]
        odds_df = odds_df.drop_duplicates(subset=['EventId', 'BookName', 'MarketName'], keep='first')

        odds_pivoted = odds_df.pivot_table(index='EventId',
                                    columns=['BookName', 'MarketName'],
                                    values=['CurrentAwayPoints', 'CurrentAwayPrice', 'CurrentHomePoints', 'CurrentHomePrice', 'OpenerAwayPoints', 'OpenerAwayPrice', 'OpenerHomePoints', 'OpenerHomePrice'],
                                    aggfunc='first')

        odds_pivoted.columns = ['_'.join(col).strip() for col in odds_pivoted.columns.values]

        odds_pivoted.reset_index(inplace=True)

        oddsCombine = events_df.merge(odds_pivoted, how = 'left', left_on = 'Id', right_on = 'EventId')

        oddsCombine.to_csv('data/' + str(dateInput) + '.csv', index = False)
        time.sleep(0.5)

scrape_odds_between(dateInputs, bookIds, marketIds)
