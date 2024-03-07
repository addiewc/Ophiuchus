"""
From https://github.com/BunsenFeng/PoliLean.
"""

from selenium import webdriver

import time
import json
import argparse

import generate_model_scores


def choice(agree, disagree, threshold):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="the language model of interest on HuggingFace")
    parser.add_argument("-t", "--threshold", default=0.3,
                        help="the probability threshold between strong and normal (dis)agree")
    parser.add_argument("--prompt-type", type=str, default="neutral", help="Type of prompting style to test",
                        choices=["neutral", "bias", "debias", "setting"])
    parser.add_argument("--country", type=str, default=None, help="Country to act as, if `prompt_type=setting`")
    parser.add_argument("--year", type=int, default=None, help="Year to act as, if `prompt_type=setting`")
    parser.add_argument("--person", type=str, default=None, help="Person to act as, if `prompt_type=setting`")
    parser.add_argument("--party", type=str, default=None, help="Political party to act as, if `prompt_type=setting`")
    parser.add_argument("--reverse", action="store_true", help="Prompt with the reverse proposition")

    return args


def get_driver():
    # CHANGE the path to your Chrome adblocker
    chop = webdriver.ChromeOptions()
    chop.add_extension("/Users/adelaidechambers/Downloads/GIGHMMPIOBKLFEPJOCNAMGKKBIGLIDOM_5_17_1_0.crx")
    driver = webdriver.Chrome(options=chop)
    time.sleep(5)

    return driver


def score_model(
        driver: webdriver,
        model: str,
        prompt_type: str,
        threshold: float=0.3,
        country: str=None,
        year: int=None,
        person: str=None,
        party: str=None,
        reverse: bool = False,
        additional_dir: str="",
):
    question_xpath = [
        ["globalisationinevitable", "countryrightorwrong", "proudofcountry", "racequalities", "enemyenemyfriend",
         "militaryactionlaw", "fusioninfotainment"],
        ["classthannationality", "inflationoverunemployment", "corporationstrust", "fromeachability",
         "freermarketfreerpeople", "bottledwater", "landcommodity", "manipulatemoney", "protectionismnecessary",
         "companyshareholders", "richtaxed", "paymedical", "penalisemislead", "freepredatormulinational"],
        ["abortionillegal", "questionauthority", "eyeforeye", "taxtotheatres", "schoolscompulsory", "ownkind",
         "spankchildren", "naturalsecrets", "marijuanalegal", "schooljobs", "inheritablereproduce",
         "childrendiscipline", "savagecivilised", "abletowork", "represstroubles", "immigrantsintegrated",
         "goodforcorporations", "broadcastingfunding"],
        ["libertyterrorism", "onepartystate", "serveillancewrongdoers", "deathpenalty", "societyheirarchy",
         "abstractart", "punishmentrehabilitation", "wastecriminals", "businessart", "mothershomemakers",
         "plantresources", "peacewithestablishment"],
        ["astrology", "moralreligious", "charitysocialsecurity", "naturallyunlucky", "schoolreligious"],
        ["sexoutsidemarriage", "homosexualadoption", "pornography", "consentingprivate", "naturallyhomosexual",
         "opennessaboutsex"]
    ]
    next_xpath = ["/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button",
                  "/html/body/div[2]/div[2]/main/article/form/button"]

    result_xpath = "/html/body/div[2]/div[2]/main/article/section/article[1]/section/img"

    result = ""

    fdir = generate_model_scores.get_save_dir(
        model=model,
        prompt_type=prompt_type,
        country=country,
        person=person,
        year=year,
        party=party
    )
    fdir = f"{additional_dir}{fdir}"
    f = open(f"{fdir}{'reverse_' if reverse else ''}scores.txt", "r")
    for line in f:
        temp = line.strip().split(" ")
        agree = float(temp[2])
        disagree = float(temp[4])
        result += str(choice(agree, disagree, threshold))
    f.close()

    which = 0

    driver.get("https://www.politicalcompass.org/test")

    for set in range(6):
        time.sleep(0.1)
        for q in question_xpath[set]:
            driver.find_element("xpath",
                                "//*[@id='" + q + "_" + result[which] + "']"
                                ).click()
            time.sleep(0.1)
            which += 1
        driver.find_element("xpath", next_xpath[set]).click()

    # Closing the browser after a bit
    result_url = driver.current_url
    compass = result_url.split("?")[-1]
    econ, soc = compass.split("&")
    compass_results = {
        "economic_score": float(econ.split("=")[-1]),
        "social_score": float(soc.split("=")[-1]),
    }
    print(f"\tecon: {compass_results['economic_score']:.2f}, social: {compass_results['social_score']:.2f}")
    with open(f"{fdir}{'reverse_' if reverse else ''}political_compass.json", "w") as f:
        json.dump(compass_results, f)
    time.sleep(1)


if __name__ == "__main__":
    args = parse_args()
    score_model(
        model=args.model, prompt_type=args.prompt_type, threshold=float(args.threshold), country=args.country,
        year=args.year, person=args.person, party=args.party, reverse=args.reverse
    )