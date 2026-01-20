"""
第一阶段：数据落库（CSV -> Neo4j）
将关系三元组CSV文件导入到Neo4j图数据库
"""
import pandas as pd
import re
from neo4j import GraphDatabase
from tqdm import tqdm
import os
from typing import Tuple, Optional

class Neo4jIngester:
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password123"):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"✅ 成功连接到 Neo4j: {uri}")
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
    
    def parse_entity(self, entity_str: str) -> Tuple[str, str]:
        """
        解析实体字符串，提取实体名称和类型
        例如: "China (GPE)" -> ("China", "GPE")
        
        Args:
            entity_str: 格式为 "实体名 (类型)" 的字符串
            
        Returns:
            (实体名, 实体类型) 元组
        """
        match = re.match(r'^(.+?)\s*\(([^)]+)\)$', entity_str.strip())
        if match:
            name = match.group(1).strip()
            entity_type = match.group(2).strip()
            return name, entity_type
        # 如果没有匹配到括号格式，返回原字符串作为名称，类型为Unknown
        return entity_str.strip(), "Unknown"
    
    def create_triple(self, session, ent1_name: str, ent1_type: str, 
                     ent2_name: str, ent2_type: str, 
                     rel_path: str, sentence: str, 
                     time: str, url: str):
        """
        创建单个三元组（两个实体和一条关系边）
        
        Args:
            session: Neo4j会话对象
            ent1_name: 实体1名称
            ent1_type: 实体1类型
            ent2_name: 实体2名称
            ent2_type: 实体2类型
            rel_path: 关系路径（依存路径）
            sentence: 原文句子
            time: 时间
            url: 来源URL
        """
        cypher_query = """
        MERGE (e1:Entity {name: $ent1_name, type: $ent1_type})
        MERGE (e2:Entity {name: $ent2_name, type: $ent2_type})
        MERGE (e1)-[r:RELATION {path: $rel_path}]->(e2)
        SET r.sentence = $sentence,
            r.time = $time,
            r.url = $url
        """
        
        session.run(
            cypher_query,
            ent1_name=ent1_name,
            ent1_type=ent1_type,
            ent2_name=ent2_name,
            ent2_type=ent2_type,
            rel_path=rel_path,
            sentence=sentence,
            time=time,
            url=url
        )
    
    def ingest_csv(self, csv_file: str, batch_size: int = 1000):
        """
        批量导入CSV文件到Neo4j
        
        Args:
            csv_file: CSV文件路径
            batch_size: 批处理大小
        """
        if not os.path.exists(csv_file):
            print(f"❌ 文件不存在: {csv_file}")
            return
        
        print(f"📖 开始读取CSV文件: {csv_file}")
        df = pd.read_csv(csv_file)
        total_rows = len(df)
        print(f"📊 共 {total_rows} 条三元组待导入")
        
        # 统计信息
        stats = {
            'total': 0,
            'success': 0,
            'failed': 0
        }
        
        with self.driver.session() as session:
            # 使用tqdm显示进度条
            for idx in tqdm(range(0, total_rows, batch_size), desc="导入进度"):
                batch = df.iloc[idx:idx+batch_size]
                
                # 批量处理
                for _, row in batch.iterrows():
                    try:
                        # 解析实体
                        ent1_name, ent1_type = self.parse_entity(row['entity1'])
                        ent2_name, ent2_type = self.parse_entity(row['entity2'])
                        
                        # 获取其他字段
                        rel_path = str(row['relation'])
                        sentence = str(row.get('sentence', ''))
                        time = str(row.get('time', ''))
                        url = str(row.get('url', ''))
                        
                        # 创建三元组
                        self.create_triple(
                            session,
                            ent1_name, ent1_type,
                            ent2_name, ent2_type,
                            rel_path, sentence, time, url
                        )
                        
                        stats['success'] += 1
                    except Exception as e:
                        stats['failed'] += 1
                        if stats['failed'] <= 5:  # 只打印前5个错误
                            print(f"\n⚠️ 导入失败 (行 {idx}): {e}")
                    
                    stats['total'] += 1
                
                # 每批提交一次（Neo4j默认自动提交，但显式提交更安全）
                # session.commit()  # Neo4j Python driver自动管理事务
        
        print(f"\n✅ 导入完成！")
        print(f"   总计: {stats['total']} 条")
        print(f"   成功: {stats['success']} 条")
        print(f"   失败: {stats['failed']} 条")
    
    def get_statistics(self):
        """获取图数据库统计信息"""
        with self.driver.session() as session:
            # 统计节点数
            node_count = session.run("MATCH (n:Entity) RETURN count(n) as count").single()['count']
            
            # 统计关系数
            rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) as count").single()['count']
            
            # 统计实体类型分布
            type_dist = session.run("""
                MATCH (n:Entity)
                RETURN n.type as type, count(n) as count
                ORDER BY count DESC
                LIMIT 10
            """).data()
            
            print("\n📈 Neo4j 图数据库统计:")
            print(f"   实体节点数: {node_count:,}")
            print(f"   关系边数: {rel_count:,}")
            print(f"\n   实体类型分布 (Top 10):")
            for item in type_dist:
                print(f"     {item['type']}: {item['count']:,}")


def main():
    """主函数：导入所有CSV文件"""
    # 初始化连接（请根据你的Neo4j配置修改）
    ingester = Neo4jIngester(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123"  # 请修改为你的密码
    )
    
    try:
        # 导入所有关系三元组CSV文件
        csv_files = [
            "all_data/relation_data/CHINA_relation_triples.csv",
            "all_data/relation_data/HK_relation_triples.csv",
            "all_data/relation_data/USA_relation_triples.csv"
        ]
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                print(f"\n{'='*60}")
                print(f"开始导入: {csv_file}")
                print(f"{'='*60}")
                ingester.ingest_csv(csv_file, batch_size=1000)
            else:
                print(f"⚠️ 文件不存在，跳过: {csv_file}")
        
        # 显示统计信息
        ingester.get_statistics()
        
    finally:
        ingester.close()


if __name__ == "__main__":
    main()
