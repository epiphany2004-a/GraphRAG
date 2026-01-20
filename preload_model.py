"""
模型预加载脚本
"""
import os
import logging
from graph_rag_retriever import _get_cached_embedding_model, clear_embedding_model_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preload_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Args:
        model_name: 要预加载的模型名称
    """
    logger.info("=" * 60)
    logger.info("开始预加载模型...")
    logger.info("=" * 60)
    
    try:
        model = _get_cached_embedding_model(
            model_name=model_name,
            device="cpu",
            trust_remote_code=False,
        )
        logger.info("=" * 60)
        logger.info("模型预加载完成")
        logger.info("=" * 60)
        return model
    except Exception as e:
        logger.error(f"模型预加载失败: {e}")
        raise


def main():
    """预加载默认模型"""
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "sentence-transformers/all-MiniLM-L6-v2"
    
    logger.info(f"预加载模型: {model_name}")
    preload_embedding_model(model_name)
    
    logger.info("\n模型已加载到内存缓存中")
    logger.info("按 Ctrl+C 退出")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n退出预加载脚本")


if __name__ == "__main__":
    main()
