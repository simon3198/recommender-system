# url 뒤에 "daily" 대신 "-monthly" 붙여야함
from setting import category_dict
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


if __name__=="__main__":

    query_dict={}
    channel_id_dict={}

    # for key,value in list(category_dict.items()):
    #     category_list.append(key)
    #     value = value.replace('daily','montly')
    #     url_list.append(value)
    #     print(key,value)
    for key,value in list(category_dict.items()):
        query_dict[key] = []
        channel_id_dict[key]={}
        for text in value:
            if len(text)==0:
                continue
            temp = text.split('#')[0].split(' ')
            # print(temp)
            result = extract_non_numeric_substring(temp[0])
            temp[0] = result

            # print(temp[:-1])
            query = ' '.join(temp[:-1])
            query = query.replace('NEW','')
            # print(query)

            # key 는 카테고리 value 는 검색해야할 검색명
            query_dict[key].append(query)
        channel_id_dict[key] = {}
        # break
            
    # print(query_dict['먹방'])

    keys = list(category_dict.keys())

    for key in keys:

        for q in query_dict[key]:
            
            search_url = f"http://localhost:8080/YouTube-operational-API/search?part=snippet&maxResults=1&q={q}"

            response = requests.get(search_url)

            response = response.json()

            channel_id = response['items'][0]['snippet']['channelId']
            if channel_id ==None:
                continue
            channel_id_dict[key][q] = channel_id
            print(q,channel_id)

        with open(f"files/{key}_channelid.json", "w", encoding='UTF-8') as json_file:

            json.dump(channel_id_dict[key], json_file, ensure_ascii=False)
    # data ={
    #     "한글" : '한글'
    # }
    # key = '123'
    # with open(f"files/{key}_channelid.json", "w", encoding='UTF-8') as json_file:
    #     json.dump(data, json_file, ensure_ascii=False)

