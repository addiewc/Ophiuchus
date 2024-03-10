import argparse
import json
import time
from pprint import pprint

import step3_testing

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", type=str, help="Name of model for fine-tuning.")
parser.add_argument("-m", "--model", type=str, help="Model to use.")
parser.add_argument("--reverse", action="store_true", help="Prompt with the reverse propositions")
parser.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="the probability threshold between strong and normal (dis)agree")
args = parser.parse_args()


driver = step3_testing.get_driver()
personas = ["Clinton", "Trump"] if "usa" in args.name else ["Green", "Labour", "National", "NZ_First"]

# generate responses and score for each prompt.
for persona in personas:
    print("\n-----------Starting:")
    pprint(persona)

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

    fdir = f"LM_results/finetuned_{args.name}_{args.model}/{persona}/"
    f = open(f"{fdir}{'reverse_' if args.reverse else ''}scores.txt", "r")
    for line in f:
        temp = line.strip().split(" ")
        agree = float(temp[2])
        disagree = float(temp[4])
        result += str(step3_testing.choice(agree, disagree, args.threshold))
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
    with open(f"{fdir}{'reverse_' if args.reverse else ''}political_compass.json", "w") as f:
        json.dump(compass_results, f)
    time.sleep(1)

print("-----------------------------------------------------------------------")
print("All prompts completed.")
