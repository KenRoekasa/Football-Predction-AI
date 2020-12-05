from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC

from actions import Actions
from selenium import webdriver

from bs4 import BeautifulSoup

import time

from selenium.common.exceptions import NoSuchElementException
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
time.sleep(2)
dropdown_season_select = driver.find_element_by_xpath(
    "//*[@id='downshift-2929-item-1']")  # TODO change 1 for different seasons
dropdown_season_select.click()

time.sleep(2)

# games_list = driver.find_element_by_xpath("//div[@class='styles__EventListContent-b3g57w-2 dotAOs']")


# print(soup)

# games = soup.find_all('a')
match_ids = []


def getMatchIds(match_ids, page_source):
    soup = BeautifulSoup(page_source, 'lxml')
    table = soup.find_all('div', class_="styles__EventListContent-b3g57w-2 dotAOs")
    for games in table:
        matches = games.find_all("a", class_="EventCellstyles__Link-sc-1m83enb-0 dhKVQJ")
        for match in matches:
            match_ids.append(match.get('data-id'))
    return match_ids


previous = driver.find_elements_by_xpath("//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")
while len(previous) > 0:

    match_ids = getMatchIds(match_ids, driver.page_source)

    time.sleep(2)
    # press previous


    # element = driver.find_element(By(("//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")))
    # actions = ActionChains(driver)
    # actions.move_to_element(previous).click().perform()
    previous[0].click()
    time.sleep(3)
    previous = driver.find_elements_by_xpath(
        "//div[@class='Cell-decync-0 styles__EventListHeader-b3g57w-0 bSxBJT']/div[1]")


driver.close()
