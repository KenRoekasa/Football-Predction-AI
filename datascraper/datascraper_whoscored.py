import csv
import sys
import time

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm
import selenium.webdriver.support.expected_conditions as ec

DEBUG = 0  # to show debug text


def get_match_stats(link):
    stats = []
    driver.get("https://www.whoscored.com%s" % link)
    time.sleep(2)  # let the link load
    soup = BeautifulSoup(driver.page_source, 'lxml')
    date = soup.find_all('dd')[4]  # [3].find('dl').find_all('dd')
    score = soup.find_all('dd')[2]
    scores = [int(s) for s in score.text.split() if s.isdigit()]
    teams = []
    teams_link = soup.find_all('a', class_='team-link')
    for team in teams_link:
        if team.text not in teams:
            teams.append(team.text)
    stats.append(date.text)
    stats.append(link)
    stats += teams
    stats += scores
    options_header = driver.find_element_by_xpath("//*[@id='live-chart-stats-options']")
    options = options_header.find_elements_by_tag_name('li')

    match_report_stat = []

    match_report_stat = extract_report_data(match_report_stat, 'live-goals-info',
                                            "//div[@id='live-goals-content']//div[@class='stat']")

    # gather passing data
    options[1].click()

    wait = WebDriverWait(driver,10)
    wait.until(ec.visibility_of_all_elements_located((By.XPATH,"//div[@id='live-passes-content']//div[@class='stat']")))

    # Get box for live score content to click on
    match_report_stat = extract_report_data(match_report_stat, 'live-passes-info',
                                            "//div[@id='live-passes-content']//div[@class='stat']")

    # gather aggression data
    options[2].click()

    # wait = WebDriverWait(driver, 10)
    # wait.until(ec.presence_of_all_elements_located((By.ID, "live-aggression")))

    time.sleep(1)

    soup = BeautifulSoup(driver.page_source, 'lxml')
    # print(soup.prettify())
    aggression_div = soup.find(id='live-aggression')
    for stat in aggression_div.find_all('div', class_='stat'):
        for span in stat.find_all('span', class_='stat-value'):
            match_report_stat.append(span.text)

    stats += match_report_stat
    # Match centre data collection
    match_centre_button = driver.find_element_by_xpath("//*[@id='sub-navigation']/ul/li[4]")
    match_centre_button.click()

    time.sleep(1)
    # wait = WebDriverWait(driver, 10)
    # wait.until(ec.invisibility_of_element_located((By.XPATH,"//*[@class='match-centre-stat has-stats selected']")))

    # Total shots get data
    # total_shots_more_button = driver.find_element_by_xpath("//*[@id='match-centre-stats']/div[1]/ul[1]/li[2]/div[2]")
    # total_shots_more_button.click()

    soup = BeautifulSoup(driver.page_source, 'lxml')
    stat_box = soup.find_all('li', class_='match-centre-stat match-centre-sub-stat')
    match_centre_stat = []
    for s in stat_box:
        for p in s.find_all('span', class_="match-centre-stat-value"):
            match_centre_stat.append(p.text)
    match_centre_stat = match_centre_stat[:62]

    if DEBUG:
        print("match centre stats: " + str(match_centre_stat))
    stats += match_centre_stat
    return stats


def extract_report_data(match_report_stat, info_panel_id, clickable_xpath):
    # Get the page and now scrap the data from the bottom panel
    soup = BeautifulSoup(driver.page_source, 'lxml')
    info_div = soup.find('div', id=info_panel_id)
    for stat in info_div.find_all('div', class_='stat'):
        for span in stat.find_all('span', class_='stat-value'):
            match_report_stat.append(span.text)

    clickable_fields = driver.find_elements_by_xpath(clickable_xpath)
    for clickable_stat in clickable_fields:  # click on every stat for more details
        clickable_stat.click()

        wait = WebDriverWait(driver, 10)
        wait.until(ec.visibility_of_all_elements_located((By.ID, info_panel_id)))



        # Get the page and now scrap the data from the bottom panel
        soup = BeautifulSoup(driver.page_source, 'lxml')
        info_div = soup.find('div', id=info_panel_id)
        for stat in info_div.find_all('div', class_='stat'):
            for span in stat.find_all('span', class_='stat-value'):
                match_report_stat.append(span.text)
    return match_report_stat


if len(sys.argv) == 3:
    # Parameters to write the data to
    links_csv = sys.argv[1]  # "data/whoscored/match-links.csv"
    filepath = sys.argv[2]

    # leaguename = 'premierleague'
    # filepath = 'data/whoscored/%s-%d.csv' % (leaguename, 20182019)

    chromepath = "chromedriver.exe"
    driver = webdriver.Chrome(chromepath)
    driver.maximize_window()

    # accept cookies once
    driver.get("https://www.whoscored.com/")

    wait = WebDriverWait(driver, 10)
    wait.until(ec.element_to_be_clickable((By.XPATH, "//div[@class='qc-cmp2-summary-buttons']/button[2]")))


    # accept cookies
    cookie_button = driver.find_element_by_xpath("//div[@class='qc-cmp2-summary-buttons']/button[2]")
    cookie_button.click()

    match_links = []

    # Load match links
    with open(links_csv, 'r+', newline='') as myfile:
        csv_reader = csv.reader(myfile, delimiter=',')
        for row in csv_reader:
            match_links = row

    print(len(match_links))

    matches = []
    all_match_stats = []
    columns = ["date", "link", "home team", "away team", "home score", "away score",
               "home total shots", "away total shots", "home total goals", "away total goals",
               "home total conversion rate", "away total conversion rate",
               "home open play shots",
               "away open play shots", "home open play goals", "away open play goals",
               "home open play conversion rate", "away open play conversion rate",
               "home set piece shots", "away set piece shots", "home set piece goals",
               "away set piece goals", "home set piece conversion", "away set piece conversion",
               "home counter attack shots", "away counter attack shots",
               "home counter attack goals",
               "away counter attack goals", "home counter attack conversion",
               "away counter attack conversion", "home penalty shots", "away penalty shots",
               "home penalty goals", "away penalty goals", "home penalty conversion",
               "away penalty conversion", "home own goals shots", "away own goals shots",
               "home own goals goals", "away own goals goals", "home own goals conversion",
               "away own goals conversion", "home total passes", "away total passes",
               "home total average pass streak", "away total average pass streak",
               "home crosses",
               "away crosses", "home crosses average pass streak",
               "away crosses average pass streak", "home through balls", "away through balls",
               "home through balls average streak", "away through balls average streak",
               "home long balls", "away long balls", "home long balls average streak",
               "away long balls average streak", "home short  passes", "away short  passes",
               "home short passes average streak", "away short passes average streak",
               "home cards",
               "away cards", "home fouls", "away fouls", "home unprofessional",
               "away unprofessional", "home dive", "away dive", "home other", "away other",
               "home red cards", "away red cards", "home yellow cards", "away yellow cards",
               "home cards per foul", "away cards per foul", "home fouls", "away fouls",
               "home total shots", "away total shots", "home woodwork", "away woodwork",
               "home shots on target", "away shots on target", "home shots off target",
               "away shots off target", "home shots blocked", "away shots blocked",
               "home possession", "away possession", "home touches", "away touches",
               "home passes success", "away passes success", "home total passes",
               "away total passes", "home accurate passes", "away accurate passes",
               "home key passes", "away key passes", "home dribbles won", "away dribbles won",
               "home dribbles attempted", "away dribbles attempted", "home dribbled past",
               "away dribbled past", "home dribble success", "away dribble success",
               "home aerials won", "away aerials won", "home aerials won%", "away aerials won%",
               "home offensive aerials", "away offensive aerials", "home defensive aerials",
               "away defensive aerials", "home successful tackles", "away successful tackles",
               "home tackles attempted", "away tackles attempted", "home was dribbled",
               "away was dribbled", "home tackles success %", "away tackles success %",
               "home clearances", "away clearances", "home interceptions", "away interceptions",
               "home corners", "away corners", "home corner accuracy", "away corner accuracy",
               "home dispossessed", "away dispossessed", "home errors", "away errors",
               "home fouls", "away fouls", "home offsides", "away offsides"
               ]
    temp_match_links = match_links.copy()
    for link in tqdm(match_links):


        stats = get_match_stats(link)
        # with open(filepath, 'a+', newline='') as myfile:
        #     wr = csv.writer(myfile, delimiter=',')
        #     wr.writerow(stats)


        while len(stats) != len(columns):
            stats.append(np.nan)

        all_match_stats.append(stats)


        df = pd.DataFrame(all_match_stats,
                          columns=columns)

        df.to_csv(filepath, index=False)

        # remove link from list
        temp_match_links.remove(link)

        # write link to csv file

        with open(links_csv, 'w+', newline='') as myfile:
            wr = csv.writer(myfile, delimiter=',')
            wr.writerow(temp_match_links)
        # print(all_match_stats)

    driver.close()

else:
    print("Invalid arguments datascraper_whoscored.py [match link csv file] [final csv file]")

# print(matches)
