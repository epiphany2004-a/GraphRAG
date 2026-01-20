"""
OpenAI SDK 与 MCP 服务集成
"""
from fastmcp.client import Client
import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class MCPOpenAIClient:
    """
    集成 OpenAI SDK 和 MCP 服务的客户端
    """
    
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str = "https://api.deepseek.com",
        openai_model: str = "deepseek-chat",
        mcp_server_path: str = "mcp_server"
    ):
        """
        Args:
            openai_api_key: OpenAI API Key
            openai_base_url: OpenAI API Base URL
            openai_model: 使用的模型名称
            mcp_server_path: MCP 服务器模块路径（如 "mcp_server"）
        """
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.model = openai_model
        self.mcp_server_path = mcp_server_path
        self.mcp_client: Optional[Client] = None
        self._mcp_tools: Optional[List] = None
    
    async def _ensure_mcp_connected(self):
        """确保 MCP 客户端已连接"""
        if self.mcp_client is None:
            # 直接导入 MCP 服务器模块并使用 FastMCP 实例
            # 这样可以使用内存传输，避免子进程通信问题
            try:
                import importlib
                mcp_module = importlib.import_module(self.mcp_server_path)
                mcp_server = mcp_module.mcp
                # 直接传入 FastMCP 实例，使用内存传输
                self.mcp_client = Client(mcp_server)
                await self.mcp_client.__aenter__()
            except ImportError as e:
                raise ImportError(f"无法导入 MCP 服务器模块 '{self.mcp_server_path}': {e}")
            except AttributeError as e:
                raise AttributeError(f"MCP 服务器模块 '{self.mcp_server_path}' 中没有找到 'mcp' 实例: {e}")
    
    async def prepare_tools(self) -> List[Dict[str, Any]]:
        """
        从 MCP 服务获取工具并转换为 OpenAI 函数调用格式
        
        Returns:
            OpenAI 函数调用格式的工具列表
        """
        await self._ensure_mcp_connected()
        
        # 获取 MCP 工具列表
        tools_response = await self.mcp_client.list_tools()
        
        # 处理不同的返回格式
        if isinstance(tools_response, list):
            # 如果直接返回列表
            tools = tools_response
        elif hasattr(tools_response, 'tools'):
            # 如果返回的是包含 .tools 属性的对象
            tools = tools_response.tools
        else:
            # 尝试其他可能的属性
            tools = getattr(tools_response, 'tools', [])
        
        if not tools:
            raise ValueError("未找到任何 MCP 工具")
        
        # 转换为 OpenAI 函数调用格式
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {}
                }
            }
            openai_tools.append(openai_tool)
        
        self._mcp_tools = tools
        return openai_tools
    
    async def call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        调用 MCP 工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具执行结果的字符串表示
        """
        await self._ensure_mcp_connected()
        
        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            
            # 解析结果
            if result.content:
                result_text = result.content[0].text
                # 尝试解析为 JSON，如果是 JSON 则格式化，否则直接返回
                try:
                    result_data = json.loads(result_text)
                    return json.dumps(result_data, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    return result_text
            else:
                return "工具执行完成，但未返回内容"
        except Exception as e:
            return f"工具执行失败: {str(e)}"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5
    ) -> str:
        """
        与 OpenAI 模型对话，支持函数调用
        
        Args:
            messages: 对话消息列表
            tools: 工具列表（如果为 None，则从 MCP 获取）
            max_iterations: 最大迭代次数（处理函数调用的循环）
        
        Returns:
            最终回复内容
        """
        # 如果没有提供工具，从 MCP 获取
        if tools is None:
            tools = await self.prepare_tools()
        
        current_messages = messages.copy()
        iteration = 0
        
        while iteration < max_iterations:
            # 调用 OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                tools=tools if tools else None,
                stream=False
            )
            
            message = response.choices[0].message
            
            # 构建 assistant 消息
            assistant_message = {
                "role": message.role,
                "content": message.content or ""
            }
            
            # 如果有 tool_calls，需要包含在消息中
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            
            current_messages.append(assistant_message)
            
            # 检查是否有函数调用
            if message.tool_calls:
                # 处理所有函数调用
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # 调用 MCP 工具
                    tool_result = await self.call_mcp_tool(function_name, function_args)
                    
                    # 将工具结果添加到消息历史（tool 消息格式）
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                iteration += 1
                continue
            else:
                # 没有函数调用，返回最终回复
                return message.content or ""
        
        # 达到最大迭代次数
        return "达到最大迭代次数，请检查函数调用是否正确"

    async def chat_with_history(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_iterations: int = 5,
    ) -> (str, List[Dict[str, Any]]):
        """
        支持多轮对话的接口：
        - 传入完整的消息历史（包含 system / user / assistant / tool 等）
        - 返回本轮助手回复，以及更新后的消息历史
        """
        # 如果没有提供工具，从 MCP 获取（只获取一次可以在外部缓存）
        if tools is None:
            tools = await self.prepare_tools()

        current_messages = messages.copy()
        iteration = 0

        while iteration < max_iterations:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=current_messages,
                tools=tools if tools else None,
                stream=False,
            )

            message = response.choices[0].message

            assistant_message: Dict[str, Any] = {
                "role": message.role,
                "content": message.content or "",
            }

            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            current_messages.append(assistant_message)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    tool_result = await self.call_mcp_tool(function_name, function_args)

                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result,
                        }
                    )

                iteration += 1
                continue
            else:
                # 返回本轮助手回复以及更新后的消息历史
                return message.content or "", current_messages

        # 达到最大迭代次数
        return "达到最大迭代次数，请检查函数调用是否正确", current_messages
    
    async def query(self, user_message: str, system_message: Optional[str] = None) -> str:
        """
        简化的查询接口
        
        Args:
            user_message: 用户消息
            system_message: 系统消息（可选）
        
        Returns:
            AI 回复
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        else:
            messages.append({
                "role": "system",
                "content": "你是一个专业的知识问答助手。你可以使用 graph_rag_retrieve 工具从知识图谱中检索相关信息来回答问题。将你用于tool的参数也加入到你的回答中  "
            })
        
        messages.append({"role": "user", "content": user_message})
        
        return await self.chat(messages)
    
    async def close(self):
        """关闭 MCP 连接"""
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
            except Exception:
                pass
            self.mcp_client = None


async def main():
    """命令行测试：支持多轮对话"""
    # 从环境变量或直接设置 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    
    # 创建客户端
    client = MCPOpenAIClient(
        openai_api_key=api_key,
        openai_base_url=base_url,
        openai_model=model,
        mcp_server_path="mcp_server"
    )
    
    try:
        print("=" * 60)
        print("OpenAI + MCP 集成测试（多轮对话）")
        print("=" * 60)

        # 预先准备工具（避免每轮重复获取）
        tools = await client.prepare_tools()

        # 初始化对话历史
        history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的知识问答助手。你可以使用 graph_rag_retrieve 工具从知识图谱中检索相关信息来回答问题,用英语参数传入可能的实体名称，多次调用工具，不要中文参数传入"
                    "在回答中，请用自然语言明确说明你调用了哪些工具、使用了哪些关键参数（例如 query 文本），"
                    "并基于工具返回的证据进行总结。"
                ),
            }
        ]

        while True:
            user_input = input("\n用户：").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q", "退出"}:
                print("会话结束。")
                break

            history.append({"role": "user", "content": user_input})

            print("\n正在思考并可能调用图谱工具，请稍候...")
            answer, history = await client.chat_with_history(
                history,
                tools=tools,
                max_iterations=5,
            )

            print("\n助手：")
            print(answer)

        print("\n" + "=" * 60)
        print("✅ 会话结束")
        print("=" * 60)
    
    finally:
        await client.close()


def main_sync():
    """同步版本的测试函数"""
    asyncio.run(main())


if __name__ == "__main__":
    # 使用异步版本
    asyncio.run(main())
    
    # 或者使用同步版本
    # main_sync()
