import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import selenium.webdriver.support.expected_conditions as ec
import sys

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

if __name__ == "__main__":

    # filepath = "data/whoscored/match-links.csv"
    # league_url = "https://www.whoscored.com/Regions/252/Tournaments/2/Seasons/7811/England-Premier-League"

    if len(sys.argv) == 3:
        filepath = sys.argv[1]
        league_url = sys.argv[2]


        driver = webdriver.Firefox()
        driver.maximize_window()

        driver.get(league_url)

        # accept cookies
        wait = WebDriverWait(driver, 100)
        wait.until(ec.element_to_be_clickable((By.XPATH, "//div[@class='qc-cmp2-summary-buttons']/button[2]")))

        cookie_button = driver.find_element_by_xpath("//div[@class='qc-cmp2-summary-buttons']/button[2]")
        cookie_button.click()

        match_links = []

        previous_button = driver.find_element_by_xpath("//a[@class='previous button ui-state-default rc-l is-default']")

        while previous_button is not None:
            soup = BeautifulSoup(driver.page_source, 'lxml')
            match_report_list = soup.find_all('a', class_="match-link match-report rc")

            # Get links for matches
            for match in match_report_list:
                match_links.append(match['href'])
                print(match['href'])
            try:
                previous_button = driver.find_element_by_xpath(
                    "//a[@class='previous button ui-state-default rc-l is-default']")
                previous_button.click()
            except NoSuchElementException:
                break
            wait = WebDriverWait(driver, 100)
            wait.until(ec.element_to_be_clickable((By.XPATH, "//a[@class='previous button ui-state-default rc-l is-default']")))
        driver.close()
        with open(filepath, 'w+', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerow(match_links)

    else:
        print("Invalid arguments who_scored_get_match_links.py [csv file name] [league link]")
