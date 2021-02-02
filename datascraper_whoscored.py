import csv

from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from actions import Actions
from selenium import webdriver
import urllib.request, json

from bs4 import BeautifulSoup

from jsonreader import json_parser
import time

from selenium.common.exceptions import NoSuchElementException, ElementNotVisibleException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait

chromepath = "chromedriver.exe"
# firefoxpath = "geckodriver.exe"
driver = webdriver.Chrome(chromepath)
driver.maximize_window()

# def getMatches(matches_arr, page_source):
#     soup = BeautifulSoup(page_source, 'lxml')
#     table = soup.find_all('div', class_="styles__EventListContent-b3g57w-2 dotAOs")
#     for games in table:
#         matches = games.find_all("a", class_="EventCellstyles__Link-sc-1m83enb-0 dhKVQJ")
#         for match in matches:
#             if match is not None:
#                 date = match.find('div', class_='Content-sc-1o55eay-0 gYsVZh')
#                 date = date.text
#                 teams = match.find_all('div')
#                 i = match.find_all('div', class_='Section-sc-1a7xrsb-0 EventCellstyles__Status-ni00fg-2 dPpfDG')
#                 if i[0].get('title') == 'FT':
#                     home_team = teams[7].text
#                     away_team = teams[8].text
#                     home_score = teams[12].text
#                     away_score = teams[13].text
#                     data = [date, home_team, away_team, match.get('data-id'), home_score, away_score]
#                     # print(data)
#                     matches_arr.append(data)
#     return matches_arr

# accept cookies once
driver.get("https://www.whoscored.com/Regions/252/Tournaments/2/Seasons/7811/England-Premier-League")
time.sleep(1)

# accept cookies

cookie_button = driver.find_element_by_xpath("//div[@class='qc-cmp2-summary-buttons']/button[2]")
cookie_button.click()

matches = []

previous_button = driver.find_element_by_xpath("//a[@class='previous button ui-state-default rc-l is-default']")

while previous_button is not None:
    soup = BeautifulSoup(driver.page_source, 'lxml')
    match_report_list = soup.find_all('a', class_="match-link match-report rc")
    # Get links for matches
    for match in match_report_list:
        matches.append(match['href'])
        print(match['href'])
    try:
        previous_button = driver.find_element_by_xpath("//a[@class='previous button ui-state-default rc-l is-default']")
        previous_button.click()
    except NoSuchElementException:
        break
    time.sleep(1)

for match in matches:
    driver.get("https://www.whoscored.com%s" % match)
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    date = soup.find_all('div', class_="info-block cleared") #[3].find('dl').find_all('dd')


# leagues = ["england/premier-league/17", "spain/laliga/8","germany/bundesliga/35","italy/serie-a/23","france/ligue-1/34"]  # TODO change this too


#
#
# for league in leagues:
#     print(league)
#     time.sleep(1)
#     url = "https://www.sofascore.com/tournament/football/%s" % league
#     driver.get(url)
#     # driver.get("https://www.sofascore.com/tournament/football/england/premier-league/17")
#     # driver.get("https://www.sofascore.com/tournament/football/spain/laliga/8")


# for i in range(2, 7):
#
#     season_select = driver.find_element_by_xpath("//button[@class='styles__Selector-cdd802-4 iDNquT']")
#
#     season_select.click()
#     # time.sleep(2)
#
#
#     dropdown_season_select = driver.find_element_by_xpath(
#         "//div[@class='styles__MenuWrapper-cdd802-1 iDYonh']//li[%d]" % i)
#
#     season = dropdown_season_select.text
#     print(i)
#     dropdown_season_select.click()
#
#     # time.sleep(2)
#
#     # games_list = driver.find_element_by_xpath("//div[@class='styles__EventListContent-b3g57w-2 dotAOs']")
#
#     # print(soup)
#
#     # games = soup.find_all('a')
#     matches = []
#
#     previous = driver.find_elements_by_xpath(
#         "//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")
#     while len(previous) > 0:
#         matches = getMatches(matches, driver.page_source)
#
#         time.sleep(2)
#         # press previous
#
#         # element = driver.find_element(By(("//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")))
#         # actions = ActionChains(driver)
#         # actions.move_to_element(previous).click().perform()
#         try:
#             previous[0].click()
#         except ElementNotVisibleException:
#             break;
#         # time.sleep(3)
#         previous = driver.find_elements_by_xpath(
#             "//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")
#
#     print(len(matches))
#
#     for match in matches:
#         match += json_parser(int(match[3]))
#
#     season = season.replace("/", "-")
#     leaguename = league.replace("/", "-")
#
#     filepath = 'data/%s-%s.csv' % (leaguename, i)
#
#     with open(filepath, 'w+', newline='') as myfile:
#         wr = csv.writer(myfile, delimiter=',')
#         wr.writerows(matches)

driver.close()

# print(matches)
