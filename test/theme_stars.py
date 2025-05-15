import re
import requests
from pandas import read_csv
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from pathlib import Path
import traceback

URL_PATTERN = re.compile(
    r"""<a href="(.*?)" rel="external nofollow noreferrer" class="plugin-name" target="_blank">(.*?)<i class="fa fa-external-link"></i></a>"""
)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.7103.48 Safari/537.36"


def get_github_update_time(repo_url, retry=0):
    # Convert GitHub URL to API endpoint
    api_url = repo_url.replace("https://github.com/", "https://api.github.com/repos/")
    if api_url.endswith("/"):
        api_url = api_url[:-1]
    if api_url.endswith(".git"):
        api_url = api_url[:-4]

    while retry >= 0:
        response = requests.get(api_url, headers={"User-Agent": UA})
        if response.status_code == 200:
            return response.json()["stargazers_count"]
        elif response.status_code == 403:
            # print("API rate limit exceeded, waiting...")
            # sleep(10)
            pass
        elif response.status_code == 404:
            print("Project Not found")
            return 0
        retry -= 1
    return None

    # return None


def theme_handler(url):
    try:
        assert url.startswith("https://github.com/")
        # get the latest update date of the theme
        update_date = get_github_update_time(url)
        return {"url": url, "stars": update_date}
    except Exception:
        print(traceback.format_exc())
        return {}


if __name__ == "__main__":
    root = Path(__file__).parent.resolve()
    out_csv = root / "themes.csv"
    result = None

    df = read_csv(out_csv)
    # select stars is null
    result = [i[1] for i in df[df["stars"] == -1].values.tolist()]

    themes = []
    NUM_THREADS = 32

    # for i in result:
    #     r = theme_handler(i)
    #     pass

    with ThreadPool(processes=NUM_THREADS) as pool:
        results = pool.imap(theme_handler, result)
        for res in tqdm(results, total=len(result)):
            if len(res) > 0:
                themes.append(res)

    df["stars"] = df["stars"].astype("int")
    if themes:
        for row in themes:
            df.loc[df["url"] == row["url"], "stars"] = row["stars"]

    # finally read csv and sort by stars
    # set na to -1 in stars column
    df["stars"] = df["stars"].fillna(-1)
    df["stars"] = df["stars"].astype("int")
    df = df.sort_values(by="stars", ascending=False)
    df.to_csv(out_csv, index=False)

    pass
