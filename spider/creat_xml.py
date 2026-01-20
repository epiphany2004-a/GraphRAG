import pandas as pd
import os
import re


def creat_xml(name):
    # 创建保存文件的文件夹
    df = pd.read_csv(f"all_data/cleaned_data/cleaned_{name}.csv", encoding="utf-8")
    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        # 获取url并处理文件名
        url = row['url']
        filename = url.replace(":", "_").replace("/", "^")
        
        # 创建XML内容
        xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
                        <news block="domestic">
                        <doc id="{str(index+1).zfill(3)}">
                        <url>{url}</url>
                        <title>{row['title']}</title>
                        <date>{row['pub_time']}</date>
                        <source>{row['source']}</source>
                        <authors>{row['author']}</authors>
                        <paragraphs>{row['content']}</paragraphs>
                        </doc>
                        </news>'''
        # 保存到文件
        if not os.path.exists(f"all_data/xml_data/orginal_{name}_xml_data"):
            os.makedirs(f"all_data/xml_data/orginal_{name}_xml_data")
        with open(f"all_data/xml_data/orginal_{name}_xml_data/{filename}.xml", "w", encoding="utf-8") as f:
            f.write(xml_content)
    
def clean_data(name):
    if not os.path.exists(f"all_data/cleaned_data"):
        os.makedirs(f"all_data/cleaned_data")
    df = pd.read_csv(f"all_data/orginal_data/{name}.csv", encoding="utf-8")
    # 检查每行的列数是否与正常行一致
    normal_columns = 6  # 正常行的列数
    df_cleaned = df[df.apply(lambda row: row.count(), axis=1) == normal_columns]
    print(df_cleaned)
    df_cleaned.to_csv(f"all_data/cleaned_data/cleaned_{name}.csv", encoding="utf-8", index=False)


if __name__ == "__main__":
    name = "CHINA"
    clean_data(name)
    creat_xml(name)
