import pandas as pd
import spacy
from itertools import combinations
import os

def get_dependency_path(doc, ent1, ent2):
    """获取两个实体间的依存路径"""
    # 获取实体的根token
    ent1_root = [token for token in ent1 if token.dep_ == 'ROOT' or token.head not in ent1][0]
    ent2_root = [token for token in ent2 if token.dep_ == 'ROOT' or token.head not in ent2][0]
    
    # 获取从ent1到root的路径
    path1 = []
    token = ent1_root
    while token.dep_ != 'ROOT':
        path1.append(f"{token.dep_}←")
        token = token.head
    
    # 获取从ent2到root的路径
    path2 = []
    token = ent2_root
    while token.dep_ != 'ROOT':
        path2.append(f"→{token.dep_}")
        token = token.head
        
    # 组合完整路径
    return "".join(path1) + "ROOT" + "".join(path2)

def process_data(csv_file, output_file, batch_size=100):
    """处理数据并提取三元组关系"""
    # 加载spaCy模型
    nlp = spacy.load("en_core_web_sm")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 存储结果的列表
    results = []
    
    # 处理每一行数据
    for index, row in df.iterrows():
        sentence = row['sentence']
        doc = nlp(sentence)
        
        # 获取句子中的命名实体
        entities = [(ent.text, ent.label_, doc[ent.start:ent.end]) for ent in doc.ents]
        
        # 获取所有实体对的组合
        entity_pairs = list(combinations(entities, 2))
        
        # 处理每对实体
        for ent1, ent2 in entity_pairs:
            # 获取依存路径
            dep_path = get_dependency_path(doc, ent1[2], ent2[2])
            
            # 创建三元组
            triple = {
                'relation': dep_path,
                'entity1': f"{ent1[0]} ({ent1[1]})",
                'entity2': f"{ent2[0]} ({ent2[1]})",
                'time': row['time'],
                'url': row['url'],
                'sentence': sentence
            }
            results.append(triple)
        
        # 每处理batch_size条数据就保存一次
        if (index + 1) % batch_size == 0:
            save_results(results, output_file)
            print(f"已处理 {index + 1} 条数据")
    
    # 保存剩余的结果
    if results:
        save_results(results, output_file)
    
    print("处理完成！")

def save_results(results, output_file):
    """保存结果到文件"""
    # 转换为DataFrame
    df = pd.DataFrame(results)
    if not os.path.exists("all_data/relation_data/"):
        os.makedirs("all_data/relation_data")
    # 如果文件不存在，创建新文件并写入表头
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False, encoding='utf-8')
    else:
        # 如果文件存在，追加数据不写入表头
        df.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
    
    # 清空结果列表
    results.clear()

if __name__ == "__main__":
    name = "CHINA"
    input_file = f"all_data/annotationed_data/{name}_annotation_data.csv"
    output_file = f"all_data/relation_data/{name}_relation_triples.csv"
    process_data(input_file, output_file)
