import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import os
import re
import hanlp
import spacy
def safe_dep_extract(doc):
    try:
        deps = []
        for token in doc:
            if token.dep_ and token.head:
                deps.append((token.dep_, token.head.text))
        return deps
    except Exception as e:
        print(f"处理依存分析时出错: {e}")
        return []

def data_annotation(name,start_row=0,end_row=None):

    # 初始化 spacy pipeline

    nlp = spacy.load("en_core_web_sm")  
    
    # 创建保存文件的文件夹
    df = pd.read_csv(f"all_data/cleaned_data/cleaned_{name}.csv", encoding="utf-8")
    if end_row is None:
        end_row = len(df)
    if not os.path.exists(f"all_data/annotationed_data"):
        os.makedirs(f"all_data/annotationed_data")
        
    # 遍历DataFrame的指定行范围
    for index, row in df.iloc[start_row:end_row].iterrows():
        annotation_data = []
        try:
            # 获取url并处理文件名
            url = row['url']
            filename = url.replace(":", "_").replace("/", "^")
            
            # 对title和content进行自然语言处理
            title = row['title']
            content = row['content']
            
            # 使用spacy进行处理
            title_doc = nlp(title)
            
            # 将content按句子分割
            sentences = content.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]  # 去除空字符串
            
            # 处理每个句子
            content_annotated = []
            for sentence in sentences:
                try:
                    sentence_doc = nlp(sentence)
                    sentence_annotated = {
                        'sentence': sentence,
                        'tokens': [token.text for token in sentence_doc],
                        'pos': [token.pos_ for token in sentence_doc],
                        'ner': [(ent.text, ent.label_) for ent in sentence_doc.ents],
                        'dep': safe_dep_extract(sentence_doc)
                    }
                    content_annotated.append(sentence_annotated)
                    
                    # 添加句子标注数据到列表
                    annotation_data.append({
                        'sentence': sentence,
                        'ner': [(ent.text, ent.label_) for ent in sentence_doc.ents],
                        'dep': safe_dep_extract(sentence_doc),
                        'time': row['pub_time'],
                        'url': url,
                        'source': row['source'],
                        'author': row['author']
                    })
                except Exception as e:
                    print(f"处理句子时出错: {e}")
                    continue
            
            # 保存标题的自然语言处理结果
            title_annotationed = {
                'tokens': [token.text for token in title_doc],  # 分词结果
                'pos': [token.pos_ for token in title_doc],     # 词性标注结果
                'ner': [(ent.text, ent.label_) for ent in title_doc.ents],  # 命名实体识别结果
                'dep': safe_dep_extract(title_doc)  # 依存分析结果
            }
            
            # 添加标题标注数据到列表
            annotation_data.append({
                'sentence': title,
                'ner': [(ent.text, ent.label_) for ent in title_doc.ents],
                'dep': safe_dep_extract(title_doc),
                'time': row['pub_time'],
                'url': url,
                'source': row['source'],
                'author': row['author']
            })
            
            # 构建paragraphs内容
            paragraphs_content = ""
            for sent_data in content_annotated:
                paragraphs_content += f'''{sent_data['sentence']}.
                    <parse>
                        <tokens>{sent_data['tokens']}</tokens>
                        <pos>{sent_data['pos']}</pos>
                        <ner>{sent_data['ner']}</ner>
                        <dep>{sent_data['dep']}</dep>
                    </parse>
                '''
            
            # 创建XML内容
            xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
                            <news block="domestic">
                            <doc id="{str(index+1).zfill(3)}">
                            <url>{url}</url>
                            <title>{row['title']}
                                <parse>
                                    <tokens>{title_annotationed['tokens']}</tokens>
                                    <pos>{title_annotationed['pos']}</pos>
                                    <ner>{title_annotationed['ner']}</ner>
                                    <dep>{title_annotationed['dep']}</dep>
                                </parse>
                            </title>
                            <date>{row['pub_time']}</date>
                            <source>{row['source']}</source>
                            <authors>{row['author']}</authors>
                            <paragraphs>
                            {paragraphs_content}
                            </paragraphs>
                            </doc>
                            </news>'''
            
            # 保存到文件
            if not os.path.exists(f"all_data/xml_data/annotationed_{name}_xml_data"):
                os.makedirs(f"all_data/xml_data/annotationed_{name}_xml_data")
            with open(f"all_data/xml_data/annotationed_{name}_xml_data/{filename}.xml", "w", encoding="utf-8") as f:
                f.write(xml_content)
            
            annotation_df = pd.DataFrame(annotation_data)
            file_exists = os.path.exists(f'all_data/annotationed_data/{name}_annotation_data.csv')

            annotation_df.to_csv(f'all_data/annotationed_data/{name}_annotation_data.csv', 
                                columns=['sentence', 'ner', 'dep', 'time', 'url', 'source', 'author'],
                                index=False, 
                                encoding='utf-8',mode='a',header= not file_exists)
        except Exception as e:
            print(f"处理文档时出错: {e}")
            continue
    # 将标注数据保存到CSV文件


if __name__ == "__main__":
    name = "HK"
    data_annotation(name,0)
