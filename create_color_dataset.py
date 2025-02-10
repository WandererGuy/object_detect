import json 
import shutil 
import os 



folder = "D:\color_classify"
# with open('data.json') as f:
#     data = json.load(f)
#     for _, value in data.items():
#         folder_name = os.path.join(folder, value)
#         os.makedirs(folder_name, exist_ok=True)

mapping = {'red_to_orange': '0_red_to_orange', 
        'magenta_to_rose': '10_magenta_to_rose', 
        'rose_to_red': '11_rose_to_red', 
        'orange_to_yellow': '1_orange_to_yellow', 
        'yellow_to_chartreuse_green': '2_yellow_to_chartreuse_green', 
        'chartreuse_green_to_green': '3_chartreuse_green_to_green', 
        'green_to_spring_green': '4_green_to_spring_green', 
        'spring_green_to_cyan': '5_spring_green_to_cyan', 
        'cyan_to_azure': '6_cyan_to_azure', 
        'azure_to_blue': '7_azure_to_blue', 
        'blue_to_violet': '8_blue_to_violet', 
        'black_grey_white': '12_black_grey_white', 
        'violet_to_magenta': '9_violet_to_magenta'}

with open('data.json') as f:
    data = json.load(f)
    for key, value in data.items():
        folder_name = os.path.join(folder, mapping[value])
        shutil.copy(key, folder_name)
