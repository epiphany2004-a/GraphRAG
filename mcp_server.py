from fastmcp import FastMCP
from graph_rag_retriever import HybridGraphRetriever

mcp = FastMCP()

graph_rag_retriever = HybridGraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password123",
    top_k_entities=5,
    graph_depth=2,
    lazy_load_model=True
)

@mcp.tool()
def graph_rag_retrieve(query: str):
    """
    通过实体图谱检索相关信息
    
    Args:
        query: 查询语句或实体名称
    
    Returns:
        dict: 检索结果，包含上下文、实体、子图、元数据
    """
    return graph_rag_retriever.retrieve(query)

if __name__ == "__main__":
    mcp.run()