"""
混合检索引擎：向量搜索 + 图遍历
"""
import os
import re

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from typing import List, Dict, Any, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, QueryBundle
from llama_index.core.schema import NodeWithScore
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_CACHE = {}


def _get_cached_embedding_model(model_name: str, device: str = "cpu", trust_remote_code: bool = False):
    """
    Args:
        model_name: 模型名称
        device: 设备类型
        trust_remote_code: 是否信任远程代码
        
    Returns:
        缓存的模型实例
    """
    cache_key = f"{model_name}_{device}_{trust_remote_code}"
    
    if cache_key not in _EMBEDDING_MODEL_CACHE:
        import time
        start_time = time.time()
        logger.info(f"正在加载模型 '{model_name}'...")
        try:
            model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=trust_remote_code,
            )
            load_time = time.time() - start_time
            _EMBEDDING_MODEL_CACHE[cache_key] = model
            logger.info(f"模型 '{model_name}' 已加载（耗时 {load_time:.2f} 秒）")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    else:
        logger.info(f"复用已缓存的模型 '{model_name}'")
    
    return _EMBEDDING_MODEL_CACHE[cache_key]


def clear_embedding_model_cache():
    """清理嵌入模型缓存"""
    global _EMBEDDING_MODEL_CACHE
    count = len(_EMBEDDING_MODEL_CACHE)
    _EMBEDDING_MODEL_CACHE.clear()
    logger.info(f"已清理 {count} 个缓存的模型")


class HybridGraphRetriever:
    """混合图检索器"""
    
    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password123",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 top_k_entities: int = 20,
                 graph_depth: int = 2,
                 lazy_load_model: bool = True,
                 use_ner: bool = True):
        """
        Args:
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            embedding_model_name: 嵌入模型名称
            top_k_entities: 向量搜索返回的top K实体数
            graph_depth: 图遍历深度
            lazy_load_model: 是否懒加载模型
            use_ner: 是否使用 NER 提取关键锚点
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.top_k_entities = top_k_entities
        self.graph_depth = graph_depth
        self.embedding_model_name = embedding_model_name
        self.lazy_load_model = lazy_load_model
        self.use_ner = use_ner
        
        self._embedding_model = None
        self._nlp = None
        
        if use_ner:
            try:
                import spacy
                logger.info("NER 功能已启用")
            except ImportError:
                logger.warning("spacy 未安装，将使用简单的关键词提取")
                self.use_ner = False
        
        if not lazy_load_model:
            self._ensure_model_loaded()
        else:
            logger.info("已启用懒加载模式")
        
        self.driver = GraphDatabase.driver(
            neo4j_uri,
            auth=(neo4j_user, neo4j_password)
        )
        
        self.graph_store = Neo4jPropertyGraphStore(
            username=neo4j_user,
            password=neo4j_password,
            url=neo4j_uri,
            database="neo4j",
            refresh_schema=False,
        )
        
        logger.info("混合图检索器初始化完成")
    
    def _ensure_model_loaded(self):
        """确保模型已加载"""
        if self._embedding_model is None:
            try:
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
                os.makedirs(cache_dir, exist_ok=True)
                
                self._embedding_model = _get_cached_embedding_model(
                    model_name=self.embedding_model_name,
                    device="cpu",
                    trust_remote_code=False,
                )
                
                Settings.embed_model = self._embedding_model
            except Exception as e:
                logger.error(f"嵌入模型初始化失败: {e}")
                raise
    
    @property
    def embedding_model(self):
        """获取嵌入模型"""
        self._ensure_model_loaded()
        return self._embedding_model
    
    def _ensure_nlp_loaded(self):
        """确保 NER 模型已加载"""
        if self.use_ner and self._nlp is None:
            try:
                import spacy
                logger.info("正在加载 spacy NER 模型...")
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("spacy NER 模型加载完成")
            except Exception as e:
                logger.warning(f"spacy 模型加载失败: {e}，将使用简单提取")
                self.use_ner = False
    
    def _extract_anchors(self, query: str) -> List[str]:
        """
        Args:
            query: 查询字符串
            
        Returns:
            关键锚点列表
        """
        anchors = []
        
        numbers = re.findall(r'\d+(?:,\d+)*', query)
        anchors.extend(numbers)
        
        if self.use_ner:
            self._ensure_nlp_loaded()
            if self._nlp:
                try:
                    doc = self._nlp(query)
                    query_entities = [ent.text.lower() for ent in doc.ents]
                    anchors.extend(query_entities)
                except Exception as e:
                    logger.warning(f"NER 提取失败: {e}，使用简单提取")
        
        if not anchors or not self.use_ner:
            important_words = re.findall(r'\b[A-Za-z]{4,}\b', query)
            anchors.extend([w.lower() for w in important_words])
        
        anchors = list(set([a.strip() for a in anchors if len(a.strip()) >= 2]))
        
        return anchors
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def step1_enhanced_entity_search(self, query: str) -> List[Dict[str, Any]]:
        """实体定位：提取关键锚点，优先匹配稀有实体"""
        logger.info(f"Step 1: 关键锚点定位 - '{query}'")
        
        query_stripped = query.strip()
        if not query_stripped:
            return []
        
        anchors = self._extract_anchors(query)
        logger.info(f"提取到的关键锚点: {anchors}")
        
        search_terms = query_stripped.split()
        if query_stripped not in search_terms:
            search_terms.append(query_stripped)
        search_terms = [term for term in search_terms if len(term.strip()) >= 2]
        
        entities: List[Dict[str, Any]] = []
        seen = set()
        
        with self.driver.session() as session:
            for anchor in anchors:
                cypher = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($a)
                WITH e, COUNT { (e)--() } as degree
                RETURN e.name as name, e.type as type, elementId(e) as node_id, degree
                ORDER BY degree ASC
                LIMIT 3
                """
                result = session.run(cypher, a=anchor)
                for record in result:
                    key = (record["name"], record["type"])
                    if key not in seen:
                        seen.add(key)
                        entities.append({
                            "name": record["name"],
                            "type": record["type"],
                            "node_id": record["node_id"],
                            "degree": record["degree"],
                            "is_anchor": True
                        })
            
            logger.info(f"通过锚点找到 {len(entities)} 个实体")
            
            if len(entities) < self.top_k_entities:
                for term in search_terms:
                    if term.lower() in [a.lower() for a in anchors]:
                        continue
                    
                    cypher = """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($q)
                    WITH e, COUNT { (e)--() } as degree
                    RETURN e.name as name, e.type as type, elementId(e) as node_id, degree
                    ORDER BY degree ASC
                    LIMIT 20
                    """
                    result = session.run(cypher, q=term)
                    for record in result:
                        key = (record["name"], record["type"])
                        if key not in seen:
                            seen.add(key)
                            entities.append({
                                "name": record["name"],
                                "type": record["type"],
                                "node_id": record["node_id"],
                                "degree": record.get("degree", 999),
                                "is_anchor": False
                            })
            
            logger.info(f"总共找到 {len(entities)} 个候选实体")
        
        if entities:
            entities.sort(key=lambda x: (not x.get("is_anchor", False), x.get("degree", 999)))
            
            result_entities = [
                {
                    "name": e["name"],
                    "type": e["type"],
                    "node_id": e["node_id"]
                }
                for e in entities[:self.top_k_entities]
            ]
        else:
            result_entities = []
        
        logger.info(f"Step 1 找到 {len(result_entities)} 个相关实体")
        logger.info(f"实体列表: {[e['name'] for e in result_entities]}")
        return result_entities
    
    def step1_vector_search(self, query: str) -> List[Dict[str, Any]]:
        """实体定位（兼容方法）"""
        return self.step1_enhanced_entity_search(query)
    
    def step2_enhanced_expansion(self, entity_names: List[str], query: str) -> Dict[str, Any]:
        """图扩展：在 Cypher 层面过滤边上的 sentence"""
        logger.info(f"Step 2: 属性过滤扩展 - 从 {len(entity_names)} 个实体出发")
        
        numbers = re.findall(r'\d+(?:,\d+)*', query)
        important_words = re.findall(r'\b[A-Za-z]{4,}\b', query.lower())
        keywords = list(set(numbers + important_words))
        
        logger.info(f"用于过滤的关键词: {keywords}")

        with self.driver.session() as session:
            if keywords:
                cypher_query = """
                MATCH (start:Entity)-[r:RELATION]-(end:Entity)
                WHERE (start.name IN $entity_names OR end.name IN $entity_names)
                AND ANY(word IN $keywords WHERE toLower(COALESCE(r.sentence, '')) CONTAINS toLower(word))
                RETURN DISTINCT
                    start.name as start_name,
                    end.name as end_name,
                    r.path as path,
                    r.sentence as sentence,
                    r.time as time,
                    r.url as url
                ORDER BY r.time ASC
                LIMIT 50
                """
                result = session.run(cypher_query, 
                                   entity_names=entity_names,
                                   keywords=keywords)
            else:
                cypher_query = """
                MATCH (start:Entity)-[r:RELATION]-(end:Entity)
                WHERE start.name IN $entity_names OR end.name IN $entity_names
                RETURN DISTINCT
                    start.name as start_name,
                    end.name as end_name,
                    r.path as path,
                    r.sentence as sentence,
                    r.time as time,
                    r.url as url
                ORDER BY r.time ASC
                LIMIT 50
                """
                result = session.run(cypher_query, entity_names=entity_names)

            subgraph: Dict[str, Any] = {"nodes": set(), "edges": []}
            for record in result:
                subgraph["nodes"].add((record["start_name"], ""))
                subgraph["nodes"].add((record["end_name"], ""))
                subgraph["edges"].append(
                    {
                        "from": record["start_name"],
                        "to": record["end_name"],
                        "path": record["path"],
                        "sentence": record["sentence"],
                        "time": record["time"],
                        "url": record["url"],
                    }
                )

            logger.info(f"子图包含 {len(subgraph['nodes'])} 个节点, {len(subgraph['edges'])} 条边")
            return subgraph
    
    def step2_graph_expansion(self, entity_names: List[str], query: Optional[str] = None) -> Dict[str, Any]:
        """
        Args:
            entity_names: 实体名称列表
            query: 查询字符串（可选）
        """
        if query:
            return self.step2_enhanced_expansion(entity_names, query)
        else:
            logger.warning("建议使用 step2_enhanced_expansion 方法并传入 query 参数")
            with self.driver.session() as session:
                cypher_query = """
                MATCH (start:Entity)-[r:RELATION]-(end:Entity)
                WHERE start.name IN $entity_names OR end.name IN $entity_names
                RETURN DISTINCT
                    start.name as start_name,
                    end.name as end_name,
                    r.path as path,
                    r.sentence as sentence,
                    r.time as time,
                    r.url as url
                ORDER BY r.time ASC
                LIMIT 50
                """
                result = session.run(cypher_query, entity_names=entity_names)
                subgraph: Dict[str, Any] = {"nodes": set(), "edges": []}
                for record in result:
                    subgraph["nodes"].add((record["start_name"], ""))
                    subgraph["nodes"].add((record["end_name"], ""))
                    subgraph["edges"].append({
                        "from": record["start_name"],
                        "to": record["end_name"],
                        "path": record["path"],
                        "sentence": record["sentence"],
                        "time": record["time"],
                        "url": record["url"],
                    })
                return subgraph
    
    def step3_format_context(self, entities: List[Dict], subgraph: Dict[str, Any], query: str) -> str:
        """
        增强版 Step 3：对边上的原文进行“证据链重排”，优先输出与型号和时间强相关的句子
        """
        logger.info("Step 3: 证据链重排")

        query_keywords = set(re.findall(r"[A-Za-z0-9]+", query.lower()))

        edges = subgraph.get("edges", [])

        # 对所有找到的句子进行“关联度打分”
        scored_sentences = []
        for edge in edges:
            sentence = edge.get("sentence") or ""
            sent_lower = sentence.lower()
            if not sent_lower:
                continue

            score = 0
            for kw in query_keywords:
                if kw and kw in sent_lower:
                    score += 10


            scored_sentences.append((score, edge))

        grouped: Dict[tuple, Dict[str, Any]] = {}
        for score, edge in scored_sentences:
            sentence = (edge.get("sentence") or "").strip()
            time_val = (edge.get("time") or "").strip()
            url_val = (edge.get("url") or "").strip()
            key = (sentence, time_val, url_val)

            fact = f"{edge['from']} --({edge['path']})--> {edge['to']}"
            if key not in grouped:
                grouped[key] = {
                    "score": score,
                    "sentence": sentence,
                    "time": time_val,
                    "url": url_val,
                    "facts": [fact],
                }
            else:
                grouped[key]["score"] = max(grouped[key]["score"], score)
                if fact not in grouped[key]["facts"]:
                    grouped[key]["facts"].append(fact)

        grouped_items = list(grouped.values())
        grouped_items.sort(key=lambda x: (x["score"], len(x["facts"])), reverse=True)

        context_parts: List[str] = []
        context_parts.append("## 检索到的核心证据（按关联度排序）")

        for i, item in enumerate(grouped_items[:15], 1):
            context_parts.append(f"{i}. [原文]: {item['sentence']}")
            context_parts.append(f"   [时间]: {item.get('time', '')} | [来源]: {item.get('url', '')}")
            context_parts.append("")

        context_text = "\n".join(context_parts)
        logger.info(f"上下文长度: {len(context_text)} 字符")

        return context_text
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Args:
            query: 用户查询问题
            
        Returns:
            检索结果，包含上下文文本和元数据
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始混合检索: '{query}'")
        logger.info(f"{'='*60}")
        
        entities = self.step1_vector_search(query)
        
        if not entities:
            logger.warning("未找到相关实体")
            return {
                'context': "未找到相关信息",
                'entities': [],
                'subgraph': {},
                'metadata': {
                    'query': query,
                    'entities_found': 0,
                    'subgraph_nodes': 0,
                    'subgraph_edges': 0
                }
            }
        
        entity_names = [e['name'] for e in entities]
        subgraph = self.step2_enhanced_expansion(entity_names, query)
        context = self.step3_format_context(entities, subgraph, query)
        
        result = {
            'context': context,
            'entities': entities,
            'subgraph': {
                'nodes': list(subgraph['nodes']),
                'edges_count': len(subgraph['edges'])
            },
            'metadata': {
                'query': query,
                'entities_found': len(entities),
                'subgraph_nodes': len(subgraph['nodes']),
                'subgraph_edges': len(subgraph['edges'])
            }
        }
        
        logger.info(f"检索完成")
        logger.info(f"找到实体: {len(entities)} 个")
        logger.info(f"子图节点: {len(subgraph['nodes'])} 个")
        logger.info(f"子图边: {len(subgraph['edges'])} 条")
        
        return result


def main():
    """测试混合检索"""
    retriever = HybridGraphRetriever(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password123",
        top_k_entities=10,
        graph_depth=2,
        lazy_load_model=True
    )
    
    try:
        test_queries = [
                "585,000 doses vaccine Germany to Hong Kong arrival time",
                "BioNTech Pfizer vaccine shipment Germany Hong Kong 585,000 doses",
                "vaccine shipment arrival Hong Kong Germany",
                "2021 vaccine delivery Hong Kong Germany 585,000",
        ]
        
        for query in test_queries:
            result = retriever.retrieve(query)
            print(f"\n{'='*60}")
            print(f"查询: {query}")
            print(f"{'='*60}")
            print(result['context'])
            print(f"\n元数据: {result['metadata']}")
            print("\n")
    
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
