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
driver.get("https://www.sofascore.com/tournament/football/england/premier-league/17")

cookie_button = driver.find_element_by_xpath("//*[@id='onetrust-accept-btn-handler']")

cookie_button.click()

season_select = driver.find_element_by_xpath("//button[@class='styles__Selector-cdd802-4 iDNquT']")

season_select.click()
# time.sleep(2)
dropdown_season_select = driver.find_element_by_xpath(
    "//*[@id='downshift-2929-item-1']")  # TODO change 1 for different seasons
dropdown_season_select.click()

# time.sleep(2)

# games_list = driver.find_element_by_xpath("//div[@class='styles__EventListContent-b3g57w-2 dotAOs']")


# print(soup)

# games = soup.find_all('a')
matches = []


def getMatches(matches_arr, page_source):
    soup = BeautifulSoup(page_source, 'lxml')
    table = soup.find_all('div', class_="styles__EventListContent-b3g57w-2 dotAOs")
    for games in table:
        matches = games.find_all("a", class_="EventCellstyles__Link-sc-1m83enb-0 dhKVQJ")
        for match in matches:
            if match is not None:
                date = match.find('div', class_='Content-sc-1o55eay-0 gYsVZh')
                date = date.text
                teams = match.find_all('div')
                i = match.find_all('div', class_='Section-sc-1a7xrsb-0 EventCellstyles__Status-ni00fg-2 dPpfDG')
                if i[0].get('title') == 'FT':
                    home_team = teams[7].text
                    away_team = teams[8].text

                    home_score = teams[12].text
                    away_score = teams[13].text
                    data = [date, home_team, away_team, match.get('data-id'), home_score, away_score]
                    print(data)
                    matches_arr.append(data)
    return matches_arr


previous = driver.find_elements_by_xpath("//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")
while len(previous) > 0:
    matches = getMatches(matches, driver.page_source)

    time.sleep(2)
    # press previous

    # element = driver.find_element(By(("//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")))
    # actions = ActionChains(driver)
    # actions.move_to_element(previous).click().perform()
    try:
        previous[0].click()
    except ElementNotVisibleException:
        break;
    # time.sleep(3)
    previous = driver.find_elements_by_xpath(
        "//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")

print(len(matches))
driver.close()




for match in matches:
    match += json_parser(int(match[3]))

with open('data/data.csv', 'w+', newline='') as myfile:
    wr = csv.writer(myfile, delimiter=',')
    wr.writerows(matches)
print(matches)
