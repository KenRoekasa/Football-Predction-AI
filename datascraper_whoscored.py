import csv
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import ElementNotVisibleException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm
from selenium.webdriver.support import expected_conditions as EC
from progress_bar import progress

# Parameters to write the data to

DEBUG = 0  # to show debug text

leaguename = 'premierleague'
filepath = 'data/whoscored/%s-%d.csv' % (leaguename, 20182019)

chromepath = "chromedriver.exe"
driver = webdriver.Chrome(chromepath)
driver.maximize_window()

# accept cookies once
driver.get("https://www.whoscored.com/")
driver.implicitly_wait(10)

time.sleep(1)

# accept cookies

cookie_button = driver.find_element_by_xpath("//div[@class='qc-cmp2-summary-buttons']/button[2]")
cookie_button.click()

match_links = []
match_links_csv = "data/whoscored/match-links1819.csv"
# Load match links
with open(match_links_csv, 'r+', newline='') as myfile:
    csv_reader = csv.reader(myfile, delimiter=',')
    for row in csv_reader:
        match_links = row

print(len(match_links))

matches = []
all_match_stats = []


def get_match_stats(link):
    stats = []
    driver.get("https://www.whoscored.com%s" % link)
    time.sleep(1)  # let the link load
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
    time.sleep(1)
    # Get box for live score content to click on
    match_report_stat = extract_report_data(match_report_stat, 'live-passes-info',
                                            "//div[@id='live-passes-content']//div[@class='stat']")

    # gather aggression data
    options[2].click()
    time.sleep(1)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    aggression_div = soup.find(id='live-aggression')
    for stat in aggression_div.find_all('div', class_='stat'):
        for span in stat.find_all('span', class_='stat-value'):
            match_report_stat.append(span.text)

    # # match report stat gather
    #
    # for i in range(0, 3):  # click every section of the situational report
    #     options[i].click()
    #
    #     if i == 2:
    #         break
    #
    #     time.sleep(1)  # let the new options load
    #
    #     stat_selection = driver.find_elements_by_xpath("//div[@class='stat']")
    #     soup = BeautifulSoup(driver.page_source, 'lxml')
    #
    #     if i == 0:
    #         live_goals_info_div = soup.find('div', id='live-goals-info')
    #         for stat in live_goals_info_div.find_all('div', class_='stat'):
    #             for span in stat.find_all('span', class_='stat-value'):
    #                 match_report_stat.append(span.text)
    #
    #     elif i == 1:
    #         live_passes_info_div = soup.find('div', id='live-passes-info')
    #         for stat in live_passes_info_div.find_all('div', class_='stat'):
    #             for span in stat.find_all('span', class_='stat-value'):
    #                 match_report_stat.append(span.text)
    #
    #     for stat in stat_selection:
    #         try:
    #             stat.click()
    #             soup = BeautifulSoup(driver.page_source, 'lxml')
    #
    #             if i == 0:
    #                 live_goals_info_div = soup.find('div', id='live-goals-info')
    #                 for stat in live_goals_info_div.find_all('div', class_='stat'):
    #                     for span in stat.find_all('span', class_='stat-value'):
    #                         match_report_stat.append(span.text)
    #
    #             elif i == 1:
    #                 live_passes_info_div = soup.find('div', id='live-passes-info')
    #                 for stat in live_passes_info_div.find_all('div', class_='stat'):
    #                     for span in stat.find_all('span', class_='stat-value'):
    #                         match_report_stat.append(span.text)
    #         except ElementNotVisibleException:
    #             pass
    # # time.sleep(1)
    # driver.save_screenshot('WebsiteScreenShot.png')
    # soup = BeautifulSoup(driver.page_source, 'lxml')
    #
    # live_agression_div = soup.find('div', id='live-aggression-content-comparision')
    #
    # while live_agression_div is None:
    #     soup = BeautifulSoup(driver.page_source, 'lxml')
    #     live_agression_div = soup.find('div', id='live-aggression-content-comparision')
    #
    # # print(live_agression_div)
    # for stat in live_agression_div.find_all('div', class_='stat'):
    #     for span in stat.find_all('span', class_='stat-value'):
    #         match_report_stat.append(span.text)
    # stat_selection = driver.find_elements_by_xpath("//div[@class='stat']")
    # live_aggression_info_div = soup.find('div', id='live-aggression-info')
    # for stat in live_aggression_info_div.find_all('div', class_='stat'):
    #     for span in stat.find_all('span', class_='stat-value'):
    #         match_report_stat.append(span.text)
    # if DEBUG:
    #     print("match report stat: " + str(match_report_stat))
    stats += match_report_stat
    # Match centre data collection
    match_centre_button = driver.find_element_by_xpath("//*[@id='sub-navigation']/ul/li[4]")
    match_centre_button.click()
    time.sleep(1)

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

        # Get the page and now scrap the data from the bottom panel
        soup = BeautifulSoup(driver.page_source, 'lxml')
        info_div = soup.find('div', id=info_panel_id)
        for stat in info_div.find_all('div', class_='stat'):
            for span in stat.find_all('span', class_='stat-value'):
                match_report_stat.append(span.text)
    return match_report_stat


temp_match_links = match_links.copy()
for link in tqdm(match_links):
    stats = get_match_stats(link)
    with open(filepath, 'a+', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerow(stats)

    all_match_stats.append(stats)

    # remove link from list
    temp_match_links.remove(link)

    # write link to csv file
    with open("data/whoscored/match-links.csv", 'w+', newline='') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerow(temp_match_links)
    # print(all_match_stats)

driver.close()

# print(matches)
