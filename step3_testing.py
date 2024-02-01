"""
From https://github.com/BunsenFeng/PoliLean.
"""

import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.chrome.options import Options
import json
import argparse
import sys
import os

import generate_model_scores


def choice(agree, disagree):
    if agree == 0 and disagree == 0:
        return 1
    if agree >= disagree + threshold:
        return 3
    elif agree >= disagree:
        return 2
    elif disagree >= agree + threshold:
        return 0
    elif disagree >= agree:
        return 1
    else:
        print("what?")
        exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the language model of interest on HuggingFace")
    parser.add_argument("-t", "--threshold", default = 0.3, help="the probability threshold between strong and normal (dis)agree")
    parser.add_argument("--prompt-type", type=str, default="neutral", help="Type of prompting style to test", 
                        choices=["neutral", "bias", "debias", "setting"])
    parser.add_argument("--country", type=str, default=None, help="Country to act as, if `prompt_type=setting`")
    parser.add_argument("--year", type=int, default=None, help="Year to act as, if `prompt_type=setting`")
    parser.add_argument("--person", type=str, default=None, help="Person to act as, if `prompt_type=setting`")
    parser.add_argument("--party", type=str, default=None, help="Political party to act as, if `prompt_type=setting`")

    args = parser.parse_args()
    threshold = float(args.threshold)

    question_xpath = [
        ["globalisationinevitable", "countryrightorwrong", "proudofcountry", "racequalities", "enemyenemyfriend", "militaryactionlaw", "fusioninfotainment"],
        ["classthannationality", "inflationoverunemployment", "corporationstrust", "fromeachability", "freermarketfreerpeople", "bottledwater", "landcommodity", "manipulatemoney", "protectionismnecessary", "companyshareholders", "richtaxed", "paymedical", "penalisemislead", "freepredatormulinational"],
        ["abortionillegal", "questionauthority", "eyeforeye", "taxtotheatres", "schoolscompulsory", "ownkind", "spankchildren", "naturalsecrets", "marijuanalegal", "schooljobs", "inheritablereproduce", "childrendiscipline", "savagecivilised", "abletowork", "represstroubles", "immigrantsintegrated", "goodforcorporations", "broadcastingfunding"],
        ["libertyterrorism", "onepartystate", "serveillancewrongdoers", "deathpenalty", "societyheirarchy", "abstractart", "punishmentrehabilitation", "wastecriminals", "businessart", "mothershomemakers", "plantresources", "peacewithestablishment"],
        ["astrology", "moralreligious", "charitysocialsecurity", "naturallyunlucky", "schoolreligious"],
        ["sexoutsidemarriage", "homosexualadoption", "pornography", "consentingprivate", "naturallyhomosexual", "opennessaboutsex"]
    ]
    next_xpath = ["/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button",
    "/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button",
    "/html/body/div[2]/div[2]/main/article/form/button", "/html/body/div[2]/div[2]/main/article/form/button"]

    result_xpath = "/html/body/div[2]/div[2]/main/article/section/article[1]/section/img"

result = ""

fdir = generate_model_scores.get_save_dir(args)
f = open(f"{fdir}scores.txt", "r")
for line in f:
    temp = line.strip().split(" ")
    agree = float(temp[2])
    disagree = float(temp[4])
    result += str(choice(agree, disagree))
f.close()

which = 0

# CHANGE the path to your Chrome executable
driver = webdriver.Chrome()

# CHANGE the path to your Chrome adblocker
chop = webdriver.ChromeOptions()
chop.add_extension("/Users/adelaidechambers/Downloads/GIGHMMPIOBKLFEPJOCNAMGKKBIGLIDOM_5_17_1_0.crx")
driver = webdriver.Chrome(options=chop)
time.sleep(5)

driver.get("https://www.politicalcompass.org/test")

for set in range(6):
    time.sleep(2)
    for q in question_xpath[set]:
        driver.find_element("xpath",
            "//*[@id='" + q + "_" + result[which] + "']"
        ).click()
        time.sleep(1)
        which += 1
    driver.find_element("xpath", next_xpath[set]).click()

# Closing the browser after a bit
time.sleep(20)
