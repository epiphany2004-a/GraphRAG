import pandas as pd
import os
import spacy
from creat_xml import creat_xml
from creat_xml import clean_data
import spider
from three_t import process_data

from data_annotation import data_annotation
if __name__ == "__main__":

    name_list = ["USA","HK"]
    for name in name_list[:]:
        # 爬取数据
        # spider.spider(name,1,700)
        # 清洗数据
        clean_data(name)
        # 创建xml文件
        creat_xml(name)
        # 数据标注
        data_annotation(name,1)
        # 抽取三元组
        input_file = f"all_data/annotationed_data/{name}_annotation_data.csv"
        output_file = f"all_data/relation_data/{name}_relation_triples.csv"
        process_data(input_file, output_file)
    # data_annotation(name,0)

