import json
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import scrapetube
from tqdm import tqdm

from .setting import api_key, category_dict


class Exporter:
    def __init__(self):
        pass

    def test(self):
        print("test")

    def make_dir(self):
        for key in list(category_dict.keys()):
            key = key.replace("/", "&")
            os.mkdir(f"files/video_id_scrape/{key}")

    def list_chunk(self, lst, n):
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    def get_channel_id(self):
        query_dict = {}
        channel_id_dict = {}

        for key, value in list(category_dict.items()):
            query_dict[key] = []
            channel_id_dict[key] = {}
            for text in value:
                if len(text) == 0:
                    continue
                temp = text.split("#")[0].split(" ")
                # print(temp)
                result = self.extract_non_numeric_substring(temp[0])
                temp[0] = result

                # print(temp[:-1])
                query = " ".join(temp[:-1])
                query = query.replace("NEW", "")
                # print(query)

                # key 는 카테고리 value 는 검색해야할 검색명
                query_dict[key].append(query)
            channel_id_dict[key] = {}
            # break

        # print(query_dict['먹방'])

        keys = list(category_dict.keys())

        for key in keys:
            print(key)
            # key = key.replace('/','&')

            for q in query_dict[key]:
                try:
                    search_url = f"http://localhost:8080/YouTube-operational-API/search?part=snippet&maxResults=1&q={q}"

                    response = requests.get(search_url)

                    response = response.json()

                    channel_id = response["items"][0]["snippet"]["channelId"]
                    if channel_id == None:
                        continue
                    channel_id_dict[key][q] = channel_id
                    print("success", q, channel_id)
                except:
                    print("fail", q)

            with open(
                f"files/channel_id/{key.replace('/','&')}_channelid.json",
                "w",
                encoding="UTF-8",
            ) as json_file:
                json.dump(channel_id_dict[key], json_file, ensure_ascii=False)

    def get_playlist_id(self):
        # url = f'https://www.googleapis.com/youtube/v3/playlists?part=snippet,contentDetails&channelId={channel_id}&maxResults=50&key={key}'
        keys = list(category_dict.keys())
        # print(keys)

        for k in keys:
            print(k)

            file_path = f"./files/channel_id/{k}_channelid.json"

            channel_id = None
            with open(file_path, "r") as file:
                channel_id = json.load(file)

            for key, value in list(channel_id.items()):
                search_url = f"https://www.googleapis.com/youtube/v3/playlists?part=snippet,contentDetails&channelId={value}&maxResults=20&key={api_key}"

                response = requests.get(search_url)

                response = response.json()

                with open(
                    f"files/playlist_id/{k}/{key.replace('/','&')}_playlist.json",
                    "w",
                    encoding="UTF-8",
                ) as json_file:
                    json.dump(response, json_file, ensure_ascii=False)

    def get_video_id(self):
        keys = list(category_dict.keys())

        for k in keys:
            print(k)
            path = f"./files/playlist_id/{k}/"
            # print(file_path)
            dir_list = os.listdir(path)

            print("Files and directories in '", path, "' :")

            for dirc in dir_list:
                playlist_dict = {}
                channel_name = dirc.replace("_playlist.json", "")
                print(channel_name)

                file_path = f"./files/playlist_id/{k}/{dirc}"
                with open(file_path, "r") as file:
                    playlist_id = json.load(file)

                # item 하나당 플리 하나
                items = playlist_id["items"]

                count = 0
                pid_list = []
                for item in items:
                    title = item["snippet"]["title"]
                    if "쇼츠" in title or "short" in title:
                        continue
                    pid = item["id"]
                    search_url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet,contentDetails&maxResults=15&playlistId={pid}&key={api_key}"
                    response = requests.get(search_url)

                    response = response.json()
                    if "error" in list(response.keys()):
                        print("fail", response)
                    playlist_dict[title] = response

                with open(
                    f"files/video_id/{k}/{channel_name}_video.json",
                    "w",
                    encoding="UTF-8",
                ) as json_file:
                    json.dump(playlist_dict, json_file, ensure_ascii=False)

    def video_id_cleansing(self):
        for category, value in list(category_dict.items()):
            print(category)
            path = f"./files/video_id/{category}/"
            # print(file_path)
            dir_list = os.listdir(path)

            video_id_list = []
            # 인플루언서당
            for dirc in dir_list:
                channel_name = dirc.replace("_video.json", "")
                print(channel_name)

                file_path = f"./files/video_id/{category}/{dirc}"
                with open(file_path, "r") as file:
                    data = json.load(file)

                # 플리당
                for playlist_title, playlist_detail in list(data.items()):
                    playlist_detail["items"]: list

                    # 플리에 있는 동영상 하나당
                    for video in playlist_detail["items"]:
                        video_detail_dict = {}
                        title = video["snippet"]["title"]
                        if "short" in title or "쇼츠" in title:
                            continue

                        video_id = video["contentDetails"]["videoId"]

                        video_detail_dict["video_id"] = video_id
                        video_detail_dict["publish_at"] = video["snippet"]["publishedAt"]
                        video_id_list.append(video_detail_dict)

            with open(
                f"files/video_id_refine/{category}/{channel_name}_video_refine.json",
                "w",
                encoding="UTF-8",
            ) as json_file:
                json.dump(video_id_list, json_file, ensure_ascii=False)

    def get_video_information(self):
        for category, value in tqdm(list(category_dict.items())):
            print(category)
            path = f"./files/video_id_scrape/{category}/"

            dir_list = os.listdir(path)

            # 인플루언서당
            count = 0
            for dirc in dir_list:
                influencer_video_detail = []

                channel_name = dirc.replace(".json", "")
                print(channel_name, category, count)

                file_path = f"./files/video_id_scrape/{category}/{dirc}"
                with open(file_path, "r") as file:
                    data = json.load(file)
                vid_list = []
                for video in data:
                    vid_list.append(video["vid"])

                vid_full_list = self.list_chunk(vid_list, 50)
                result = []
                for vid_chunck in vid_full_list:
                    vid = ",".join(vid_chunck)

                    # print(vid)
                    # return
                    search_url = f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={vid}&key={api_key}"
                    response = requests.get(search_url)

                    response = response.json()

                    if "error" in list(response.keys()):
                        print(category, channel_name, vid)
                        print("fail", response)
                        return

                    result.extend(response["items"])
                    # return

                with open(
                    f"files/video_information/{category}/{channel_name}_video_information.json",
                    "w",
                    encoding="UTF-8",
                ) as json_file:
                    json.dump(result, json_file, ensure_ascii=False, indent=4)
                # return

    def get_video_id_by_channelid(self):
        keys = list(category_dict.keys())

        for k in keys:
            print(k)

            file_path = f"./files/channel_id/{k}_channelid.json"

            channel_id = None
            with open(file_path, "r") as file:
                channel_id = json.load(file)
            count = 0
            for key, value in list(channel_id.items()):
                print(key)
                # if count ==100:
                #     print('-------next--------')
                #     break
                try:
                    videos = scrapetube.get_channel(value, limit=200)
                    video_list = []
                    for video in videos:
                        video_dict = {}
                        video_dict["vid"] = video["videoId"]
                        video_dict["title"] = video["title"]["runs"][0]["text"]
                        video_dict["publish"] = video["publishedTimeText"]
                        video_list.append(video_dict)
                    count += 1
                    with open(f"files/video_id_scrape/{k}/{key}.json", "w", encoding="UTF-8") as json_file:
                        json.dump(video_list, json_file, ensure_ascii=False, indent=4)
                except:
                    print(key, "fail")

    def extract_non_numeric_substring(self, input_string):
        result = ""
        found_non_numeric = False

        for char in input_string:
            if not char.isdigit():
                found_non_numeric = True
                result += char
            elif found_non_numeric:
                break

        return result

    def get_statistics(self):
        final_list = []
        col_name = [
            "channelname",
            "channelid",
            "title",
            "category",
            "publishat",
            "tags",
            "views",
            "likes",
            "comments",
        ]
        for category, value in list(category_dict.items()):
            print(category)
            path = f"./files/video_information/{category}/"

            dir_list = os.listdir(path)

            # 인플루언서당
            count = 0
            for dirc in dir_list:
                influencer_video_detail = []

                channel_name = dirc.replace("_video_information.json", "")
                # print(channel_name, category, count)

                file_path = path + f"/{dirc}"
                with open(file_path, "r") as file:
                    data = json.load(file)

                for video in data:
                    vid = video["id"]
                    snippet = video["snippet"]
                    statistics = video["statistics"]

                    # 3년 지난 영상은 취급 x
                    if datetime.fromisoformat(snippet["publishedAt"]).date() < datetime.now().date() - timedelta(
                        days=1080
                    ):
                        break
                    try:
                        final_list.append(
                            [
                                channel_name,
                                snippet["channelId"],
                                snippet["title"],
                                category,
                                datetime.fromisoformat(snippet["publishedAt"]).strftime("%Y-%m-%d"),
                                ",".join(snippet.get("tags", [" "])).replace("#", ""),
                                int(statistics["viewCount"]),
                                int(statistics["likeCount"]),
                                int(statistics["commentCount"]),
                            ]
                        )
                    except:
                        continue

        df = pd.DataFrame(final_list, columns=col_name)
        df.to_csv("./files/video_statistics.csv")
