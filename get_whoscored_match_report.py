import html
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException

chromepath = "chromedriver.exe"
# firefoxpath = "geckodriver.exe"
driver = webdriver.Chrome(chromepath)
driver.maximize_window()

driver.get(
    "https://www.whoscored.com/Matches/1376198/MatchReport/England-Premier-League-2019-2020-Sheffield-United-Everton")
time.sleep(1)
cookie_button = driver.find_element_by_xpath("//*[@id='qc-cmp2-ui']/div[2]/div/button[2]")
cookie_button.click()
time.sleep(1)
soup = BeautifulSoup(driver.page_source, 'lxml')

date = soup.find_all('dd')[4]  # [3].find('dl').find_all('dd')
score = soup.find_all('dd')[2]
scores = [int(s) for s in score.text.split() if s.isdigit()]

teams = []

teams_link = soup.find_all('a', class_='team-link')

for team in teams_link:
    if team.text not in teams:
        teams.append(team.text)

stats = []

options_header = driver.find_element_by_xpath("//*[@id='live-chart-stats-options']")

options = options_header.find_elements_by_tag_name('li')

for i in range(0, 3):
    options[i].click()

    if i == 2:
        break
    time.sleep(1)
    stat_selection = driver.find_elements_by_xpath("//div[@class='stat']")
    soup = BeautifulSoup(driver.page_source, 'lxml')
    if i == 0:
        live_goals_info_div = soup.find('div', id='live-goals-info')
        for stat in live_goals_info_div.find_all('div', class_='stat'):
            for span in stat.find_all('span', class_='stat-value'):
                stats.append(span.text)
                print(span.text)

    elif i == 1:
        live_passes_info_div = soup.find('div', id='live-passes-info')
        for stat in live_passes_info_div.find_all('div', class_='stat'):
            for span in stat.find_all('span', class_='stat-value'):
                stats.append(span.text)
                print(span.text)

    for stat in stat_selection:
        try:
            stat.click()
            soup = BeautifulSoup(driver.page_source, 'lxml')

            if i == 0:
                live_goals_info_div = soup.find('div', id='live-goals-info')
                for stat in live_goals_info_div.find_all('div', class_='stat'):
                    for span in stat.find_all('span', class_='stat-value'):
                        stats.append(span.text)
                        print(span.text)

            elif i == 1:
                live_passes_info_div = soup.find('div', id='live-passes-info')
                for stat in live_passes_info_div.find_all('div', class_='stat'):
                    for span in stat.find_all('span', class_='stat-value'):
                        stats.append(span.text)
                        print(span.text)




        except ElementNotVisibleException:
            pass
time.sleep(1)
soup = BeautifulSoup(driver.page_source, 'lxml')

stat_tag = soup.find_all('div', class_='stat-group')
for stat in stat_tag[4].find_all('div', class_='stat'):
    for span in stat.find_all('span', class_='stat-value'):
        stats.append(span.text)
        print(span.text)

stat_selection = driver.find_elements_by_xpath("//div[@class='stat']")
live_aggression_info_div = soup.find('div', id='live-aggression-info')
for stat in live_aggression_info_div.find_all('div', class_='stat'):
    for span in stat.find_all('span', class_='stat-value'):
        stats.append(span.text)
        print(span.text)

print(len(stats))

driver.close()
