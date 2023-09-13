# url 뒤에 "daily" 대신 "-monthly" 붙여야함
from setting import category_dict, api_key
from bs4 import BeautifulSoup
import requests
import re
from googleapiclient.discovery import build
import os
import json

# 플레이보드 카테고리 별로 인플루언서 리스트 수집

# 카테고리 종류 수집 먼저


def get_category_list():

    return None


# 카테고리별 인플루언서 수집

def get_influencer_list():

    return None


def get_channel_id():
    query_dict = {}
    channel_id_dict = {}

    for key, value in list(category_dict.items()):
        query_dict[key] = []
        channel_id_dict[key] = {}
        for text in value:
            if len(text) == 0:
                continue
            temp = text.split('#')[0].split(' ')
            # print(temp)
            result = extract_non_numeric_substring(temp[0])
            temp[0] = result

            # print(temp[:-1])
            query = ' '.join(temp[:-1])
            query = query.replace('NEW', '')
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

                channel_id = response['items'][0]['snippet']['channelId']
                if channel_id == None:
                    continue
                channel_id_dict[key][q] = channel_id
                print('success', q, channel_id)
            except:
                print('fail', q)

        with open(f"files/channel_id/{key.replace('/','&')}_channelid.json", "w", encoding='UTF-8') as json_file:

            json.dump(channel_id_dict[key], json_file, ensure_ascii=False)


def get_playlist_id():
    # url = f'https://www.googleapis.com/youtube/v3/playlists?part=snippet,contentDetails&channelId={channel_id}&maxResults=50&key={key}'
    keys = list(category_dict.keys())
    # print(keys)

    for k in keys[17:]:
        print(k)

        file_path = f"./files/channel_id/{k}_channelid.json"

        channel_id = None
        with open(file_path, 'r') as file:
            channel_id = json.load(file)

        for key, value in list(channel_id.items()):
            search_url = f'https://www.googleapis.com/youtube/v3/playlists?part=snippet,contentDetails&channelId={value}&maxResults=20&key={api_key}'

            response = requests.get(search_url)

            response = response.json()

            with open(f"files/playlist_id/{k}/{key.replace('/','&')}_playlist.json", "w", encoding='UTF-8') as json_file:
                json.dump(response, json_file, ensure_ascii=False)


def get_video_id():
    keys = list(category_dict.keys())

    # path 수정 필요
    # path = "./files"

    for k in keys[4:]:
        print(k)
        path = f'./files/playlist_id/{k}/'
        # print(file_path)
        dir_list = os.listdir(path)

        print("Files and directories in '", path, "' :")

        for dirc in dir_list:
            playlist_dict = {}
            channel_name = dirc.replace('_playlist.json', '')
            print(channel_name)

            file_path = f'./files/playlist_id/{k}/{dirc}'
            with open(file_path, 'r') as file:
                playlist_id = json.load(file)

            # item 하나당 플리 하나
            items = playlist_id['items']

            count = 0
            pid_list = []
            for item in items:
                title = item['snippet']['title']
                if '쇼츠' in title or 'short' in title:
                    continue
                pid = item['id']
                search_url = f'https://www.googleapis.com/youtube/v3/playlistItems?part=snippet,contentDetails&maxResults=25&playlistId={pid}&key={api_key}'
                response = requests.get(search_url)

                response = response.json()

                playlist_dict[title] = response
            # print(playlist_dict)
            with open(f"files/video_id/{k}/{channel_name}_video.json", "w", encoding='UTF-8') as json_file:
                json.dump(playlist_dict, json_file, ensure_ascii=False)


def get_video_infromation():
    # search_url = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={pid}&key={api_key}'
    # response = requests.get(search_url)

    # response = response.json()

    return None


def make_dir():
    for key in list(category_dict.keys()):
        key = key.replace('/', '&')
        os.mkdir(f'files/video_id/{key}')


def extract_non_numeric_substring(input_string):
    result = ""
    found_non_numeric = False

    for char in input_string:
        if not char.isdigit():
            found_non_numeric = True
            result += char
        elif found_non_numeric:
            break

    return result


if __name__ == "__main__":
    # make directory for key
    # make_dir()

    # get playlist id with channel id
    # get_playlist_id()

    # get video id with playlist id
    get_video_id()

    # get video information with video id
    # get_video_infromation()

    # video는 플리당 최대 10개 50개 채우면 끝
    # url = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet%2CcontentDetails%2Cstatistics&id={pid}&key={api_key}'

    # get p

    # Todo. 추출한 채널 id를 이용해서 각 채널의 동영상 정보(제목, 태그, 조회수, 좋아요, 댓글 등등 최대한 많이) 수집해서 저장
