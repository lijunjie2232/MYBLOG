import re
import requests
from datetime import datetime
from pandas import DataFrame, read_csv
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from pathlib import Path
from time import sleep
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

    api_url = api_url + "/commits"

    while retry >= 0:
        response = requests.get(api_url, headers={"User-Agent": UA})
        if response.status_code == 200:
            last_commit = response.json()[0]["commit"]["author"]["date"]
            return datetime.strptime(last_commit, "%Y-%m-%dT%H:%M:%SZ")
        elif response.status_code == 403:
            print("API rate limit exceeded, waiting...")
            # sleep(10)
        elif response.status_code == 404:
            print("Project Not found")
            return datetime.strptime("2000-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
        retry -= 1
    return None

    # return None


def theme_handler(args):
    url, name = args
    name = name.strip()
    try:
        assert url.startswith("https://github.com/")
        # get the latest update date of the theme
        update_date = get_github_update_time(url)
        return {"name": name, "url": url, "update_date": update_date}
    except:
        print(traceback.format_exc())
        return {}


if __name__ == "__main__":
    url = "https://hexo.io/themes/"
    root = Path(__file__).parent.resolve()
    out_csv = root / "themes.csv"
    result = None
    if not out_csv.is_file():
        page = requests.get(url, headers={"user-agent": UA})
        result = URL_PATTERN.findall(page.text)
    else:
        df = read_csv(out_csv)
        # select update_date is null
        result = [(i[1], i[0]) for i in df[df["update_date"].isnull()].values.tolist()]

    themes = []
    NUM_THREADS = 32

    # for i in result:
    #     r = theme_handler(i)
    #     pass

    with ThreadPool(processes=NUM_THREADS) as pool:
        results = pool.imap(theme_handler, result)
        for res in tqdm(results, total=len(result)):
            if len(res) > 0 and res["update_date"]:
                themes.append(res)
    if not out_csv.is_file():
        df = DataFrame(themes)
        df = df.sort_values(by="update_date", ascending=False)
        df.to_csv(out_csv, index=False)
    else:
        # update df
        df = read_csv(out_csv)
        # transform update_date to datetime
        df["update_date"] = df["update_date"].astype("datetime64[ns]")
        if themes:
            for row in themes:
                df.loc[df["url"] == row["url"], "update_date"] = row["update_date"]
            # df = df.sort_values(by="update_date", ascending=False)
            df.to_csv(out_csv, index=False)

    # finally read csv and sort by update_date
    df = read_csv(out_csv)
    df["update_date"] = df["update_date"].astype("datetime64[ns]")
    df = df.sort_values(by="update_date", ascending=False)
    df.to_csv(out_csv, index=False)

    pass
