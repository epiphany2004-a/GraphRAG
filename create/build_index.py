"""
第二阶段：构建 GraphRAG 索引（LlamaIndex 赋能）
为Neo4j中的节点建立向量索引，实现语义检索能力
"""
import os

# 强制 transformers 不加载 TensorFlow/Keras 路径，避免 Keras 3 与 transformers 的兼容性问题
# 我们只使用 sentence-transformers 的 PyTorch 路线即可
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from neo4j import GraphDatabase
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAGIndexBuilder:
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password123",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        初始化GraphRAG索引构建器
        
        Args:
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            embedding_model_name: HuggingFace嵌入模型名称
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # 初始化嵌入模型（使用HuggingFace免费模型）
        logger.info(f"📥 加载嵌入模型: {embedding_model_name}")
        self.embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_name,
            device="cpu"  # 如果有GPU可以改为"cuda"
        )
        
        # 配置LlamaIndex全局设置
        Settings.embed_model = self.embedding_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # 初始化Neo4j Property Graph Store
        logger.info("🔌 连接Neo4j Property Graph Store...")
        self.graph_store = Neo4jPropertyGraphStore(
            username=neo4j_user,
            password=neo4j_password,
            url=neo4j_uri,
            database="neo4j",
            # 你的 Neo4j 实例未安装 APOC 时会报 `apoc.meta.data` 不存在
            # 关闭 schema 刷新即可绕开 APOC 依赖（不影响我们后续用图数据做检索）
            refresh_schema=False,
        )
        
        logger.info("✅ GraphRAG索引构建器初始化完成")
    
    def create_node_text(self, entity_name: str, entity_type: str, 
                         relations: list) -> str:
        """
        为实体节点创建文本描述，用于向量化
        
        Args:
            entity_name: 实体名称
            entity_type: 实体类型
            relations: 与该实体相关的关系列表
            
        Returns:
            节点的文本描述
        """
        # 基础信息
        text_parts = [f"实体名称: {entity_name}", f"实体类型: {entity_type}"]
        
        # 添加关系信息
        if relations:
            text_parts.append("相关关系:")
            for rel in relations[:5]:  # 只取前5个关系，避免文本过长
                text_parts.append(f"  - {rel.get('description', '')}")
        
        return "\n".join(text_parts)
    
    def build_vector_index(self):
        """
        构建向量索引：为Neo4j中的每个实体节点创建向量表示
        """
        logger.info("🚀 开始构建向量索引...")
        
        # 连接Neo4j获取所有实体
        driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        try:
            with driver.session() as session:
                # 获取所有实体及其关系
                query = """
                MATCH (e:Entity)-[r:RELATION]->(e2:Entity)
                RETURN e.name as name, e.type as type, 
                       collect({
                           target: e2.name,
                           path: r.path,
                           sentence: r.sentence
                       }) as relations
                LIMIT 10000
                """
                
                result = session.run(query)
                nodes_data = []
                
                logger.info("📊 收集实体节点数据...")
                for record in tqdm(result, desc="收集节点"):
                    entity_name = record['name']
                    entity_type = record['type']
                    relations = record['relations']
                    
                    # 构建节点文本
                    node_text = self.create_node_text(entity_name, entity_type, relations)
                    
                    # 创建LlamaIndex节点
                    node = TextNode(
                        text=node_text,
                        metadata={
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "node_id": f"{entity_name}_{entity_type}"
                        }
                    )
                    nodes_data.append(node)
                
                logger.info(f"✅ 共收集 {len(nodes_data)} 个节点")
                
                # 批量生成嵌入向量
                logger.info("🔢 生成嵌入向量...")
                for node in tqdm(nodes_data, desc="向量化"):
                    # LlamaIndex会自动调用embed_model生成向量
                    pass
                
                # 创建向量存储上下文
                storage_context = StorageContext.from_defaults(
                    graph_store=self.graph_store
                )
                
                # 构建向量索引
                logger.info("📚 构建向量索引...")
                index = VectorStoreIndex(
                    nodes_data,
                    storage_context=storage_context,
                    show_progress=True
                )
                
                logger.info("✅ 向量索引构建完成！")
                
                # 保存索引（可选）
                # index.storage_context.persist(persist_dir="./storage")
                
                return index
                
        finally:
            driver.close()
    
    def add_embeddings_to_neo4j(self):
        """
        将生成的嵌入向量直接存储到Neo4j节点属性中
        这样可以在Cypher查询中直接使用向量相似度搜索
        """
        logger.info("💾 将嵌入向量存储到Neo4j节点...")
        
        driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        try:
            with driver.session() as session:
                # 获取所有实体
                query = "MATCH (e:Entity) RETURN e.name as name, e.type as type LIMIT 10000"
                result = session.run(query)
                
                nodes_to_process = list(result)
                logger.info(f"📊 共 {len(nodes_to_process)} 个节点需要向量化")
                
                # 批量处理
                batch_size = 100
                for i in tqdm(range(0, len(nodes_to_process), batch_size), desc="向量化节点"):
                    batch = nodes_to_process[i:i+batch_size]
                    
                    for record in batch:
                        entity_name = record['name']
                        entity_type = record['type']
                        
                        # 构建节点文本
                        node_text = f"实体名称: {entity_name}, 实体类型: {entity_type}"
                        
                        # 生成嵌入向量
                        embedding = self.embedding_model.get_text_embedding(node_text)
                        
                        # 存储到Neo4j节点属性
                        update_query = """
                        MATCH (e:Entity {name: $name, type: $type})
                        SET e.embedding = $embedding
                        """
                        session.run(
                            update_query,
                            name=entity_name,
                            type=entity_type,
                            embedding=embedding
                        )
                
                logger.info("✅ 嵌入向量已存储到Neo4j节点")
                
        finally:
            driver.close()


def main():
    """主函数"""
    # 初始化索引构建器
    builder = GraphRAGIndexBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password123",  # 请修改为你的密码
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"  # 免费模型
    )
    
    try:
        # 方法1: 构建LlamaIndex向量索引（推荐）
        index = builder.build_vector_index()
        
        # 方法2: 将向量直接存储到Neo4j节点属性（可选）
        # builder.add_embeddings_to_neo4j()
        
        logger.info("🎉 GraphRAG索引构建完成！")
        
    except Exception as e:
        logger.error(f"❌ 构建索引时出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()
