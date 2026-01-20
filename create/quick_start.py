"""
GraphRAG系统快速启动脚本
一键完成：数据导入 -> 索引构建 -> 启动服务
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_neo4j_connection():
    """检查Neo4j连接"""
    print_header("步骤0: 检查Neo4j连接")
    
    try:
        from neo4j import GraphDatabase
        
        # 从用户输入获取连接信息
        print("请输入Neo4j连接信息（直接回车使用默认值）:")
        uri = input("URI [bolt://localhost:7687]: ").strip() or "bolt://localhost:7687"
        user = input("用户名 [neo4j]: ").strip() or "neo4j"
        password = input("密码 [password123]: ").strip() or "password123"
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        
        print("✅ Neo4j连接成功！")
        return uri, user, password
        
    except Exception as e:
        print(f"❌ Neo4j连接失败: {e}")
        print("\n请确保：")
        print("1. Neo4j服务已启动")
        print("2. 连接信息正确")
        print("3. 防火墙允许连接")
        return None, None, None


def update_config_files(uri, user, password):
    """更新配置文件中的连接信息"""
    print_header("步骤0.5: 更新配置文件")
    
    files_to_update = [
        "ingest_to_neo4j.py",
        "build_index.py",
        "graph_rag_retriever.py",
        "api_server.py"
    ]
    
    for file_path in files_to_update:
        if not os.path.exists(file_path):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换连接信息（简单替换，实际应该用更robust的方法）
            # 这里只是示例，实际使用时用户需要手动修改
            print(f"⚠️  请手动修改 {file_path} 中的连接信息")
            print(f"   uri = \"{uri}\"")
            print(f"   user = \"{user}\"")
            print(f"   password = \"{password}\"")
            
        except Exception as e:
            print(f"⚠️  无法读取 {file_path}: {e}")
    
    input("\n按回车键继续（确保已修改配置文件）...")


def step1_import_data():
    """步骤1: 导入数据"""
    print_header("步骤1: 导入数据到Neo4j")
    
    csv_files = [
        "all_data/relation_data/CHINA_relation_triples.csv",
        "all_data/relation_data/HK_relation_triples.csv",
        "all_data/relation_data/USA_relation_triples.csv"
    ]
    
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ 未找到CSV文件！")
        print("   请确保以下文件存在：")
        for f in csv_files:
            print(f"   - {f}")
        return False
    
    print(f"找到 {len(existing_files)} 个CSV文件")
    
    response = input("\n是否开始导入数据？(y/n) [y]: ").strip().lower()
    if response and response != 'y':
        print("跳过数据导入")
        return True
    
    try:
        print("\n运行 ingest_to_neo4j.py...")
        result = subprocess.run(
            [sys.executable, "ingest_to_neo4j.py"],
            check=True,
            capture_output=False
        )
        print("✅ 数据导入完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 数据导入失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 找不到 ingest_to_neo4j.py 文件")
        return False


def step2_build_index():
    """步骤2: 构建索引"""
    print_header("步骤2: 构建GraphRAG索引")
    
    response = input("是否构建向量索引？(y/n) [n]: ").strip().lower()
    if response != 'y':
        print("跳过索引构建（可选步骤）")
        return True
    
    try:
        print("\n运行 build_index.py...")
        print("⚠️  首次运行会下载约80MB的嵌入模型，请耐心等待...")
        result = subprocess.run(
            [sys.executable, "build_index.py"],
            check=True,
            capture_output=False
        )
        print("✅ 索引构建完成！")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 索引构建失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ 找不到 build_index.py 文件")
        return False


def step3_start_server():
    """步骤3: 启动API服务"""
    print_header("步骤3: 启动API服务")
    
    response = input("是否启动API服务？(y/n) [y]: ").strip().lower()
    if response and response != 'y':
        print("跳过API服务启动")
        return True
    
    try:
        print("\n启动API服务器...")
        print("服务地址: http://localhost:8000")
        print("API文档: http://localhost:8000/docs")
        print("\n按 Ctrl+C 停止服务\n")
        
        subprocess.run(
            [sys.executable, "api_server.py"],
            check=False
        )
        
    except KeyboardInterrupt:
        print("\n\n✅ API服务已停止")
        return True
    except FileNotFoundError:
        print("❌ 找不到 api_server.py 文件")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("GraphRAG系统快速启动")
    print("=" * 60)
    
    print("\n本脚本将帮助您：")
    print("1. 检查Neo4j连接")
    print("2. 导入CSV数据到Neo4j")
    print("3. 构建GraphRAG索引（可选）")
    print("4. 启动API服务")
    
    # 检查Neo4j连接
    uri, user, password = check_neo4j_connection()
    if not uri:
        print("\n❌ 无法继续，请先解决Neo4j连接问题")
        return 1
    
    # 更新配置文件
    update_config_files(uri, user, password)
    
    # 步骤1: 导入数据
    if not step1_import_data():
        print("\n⚠️  数据导入失败，但可以继续其他步骤")
    
    # 步骤2: 构建索引（可选）
    step2_build_index()
    
    # 步骤3: 启动服务
    step3_start_server()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n下一步：")
    print("1. 访问 http://localhost:8000/docs 查看API文档")
    print("2. 运行 python test_graphrag.py 测试系统")
    print("3. 查看 README_GraphRAG.md 了解详细使用说明")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(1)
