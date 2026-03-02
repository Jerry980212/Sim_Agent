# -*- coding: utf-8 -*-
"""
simulation_agent_app.py — Agent + MCP 驱动的仿真对话系统
核心改进：
  1. AI Agent 理解用户意图并分步规划仿真流程
  2. 通过 MCP 协议调用标准化工具，每步有参数校验
  3. 每步执行后验证结果，失败时自动诊断和重试
  4. 消除 eventlet/threading 混用问题
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

# MCP SDK
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# LLM 调用（以 OpenAI 兼容接口为例）
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simulation-agent")

# ══════════════════════════════════════════════════════════
#  配置
# ══════════════════════════════════════════════════════════

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

# MCP Server 配置
MCP_SERVERS = {
    "pyfluent": StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(WORK_DIR, "mcp_servers", "pyfluent_mcp_server.py")],
        env={**os.environ},
    ),
    "cfd_knowledge": StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(WORK_DIR, "mcp_servers", "cfd_knowledge_mcp_server.py")],
        env={**os.environ},
    ),
}


# ══════════════════════════════════════════════════════════
#  MCP 客户端管理器
# ══════════════════════════════════════════════════════════

class MCPClientManager:
    """管理多个 MCP Server 连接，统一工具调用接口"""

    def __init__(self):
        self.sessions: dict[str, ClientSession] = {}
        self.tool_registry: dict[str, str] = {}  # tool_name -> server_name
        self.tool_schemas: list[dict] = []  # OpenAI function calling 格式

    async def connect_all(self, server_configs: dict[str, StdioServerParameters]):
        """连接所有 MCP Server 并发现工具"""
        for name, params in server_configs.items():
            try:
                await self._connect_server(name, params)
                logger.info(f"已连接 MCP Server: {name}")
            except Exception as e:
                logger.error(f"连接 MCP Server '{name}' 失败: {e}")

    async def _connect_server(self, name: str, params: StdioServerParameters):
        """连接单个 MCP Server"""
        read_stream, write_stream = await stdio_client(params).__aenter__()
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        await session.initialize()

        self.sessions[name] = session

        # 发现并注册工具
        tools_response = await session.list_tools()
        for tool in tools_response.tools:
            self.tool_registry[tool.name] = name
            # 转换为 OpenAI function calling 格式
            self.tool_schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })

        logger.info(
            f"Server '{name}' 注册了 {len(tools_response.tools)} 个工具: "
            f"{[t.name for t in tools_response.tools]}"
        )

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用指定工具"""
        server_name = self.tool_registry.get(tool_name)
        if not server_name:
            return json.dumps({"success": False, "error": f"未知工具: {tool_name}"})

        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else "{}"

    def get_openai_tools(self) -> list[dict]:
        """返回所有工具的 OpenAI function calling 格式定义"""
        return self.tool_schemas

    async def close_all(self):
        """关闭所有连接"""
        for name, session in self.sessions.items():
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════
#  AI Agent 核心
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是一个专业的 CFD 仿真工程师 AI 助手，通过 MCP 工具控制 ANSYS Fluent 进行仿真计算。

## 工作原则

1. **参数先验证**：设置任何仿真参数前，先调用 validate_simulation_parameter 或相应的知识库工具确认参数合法性
2. **分步执行**：将仿真流程分解为明确的步骤，每步执行后检查返回结果
3. **错误自动诊断**：遇到错误时，调用 diagnose_error 获取专业建议，尝试自动修复
4. **物理合理性**：始终检查参数的物理合理性（如温度不为负、速度方向正确等）
5. **状态感知**：通过 get_session_status 了解当前仿真状态，避免重复操作

## 标准仿真流程

1. launch_fluent → 启动求解器
2. read_mesh → 读取网格
3. recommend_turbulence_model → 根据应用选择模型
4. set_turbulence_model → 设置湍流模型
5. set_boundary_condition → 设置各边界条件
6. get_convergence_criteria → 获取收敛标准建议
7. run_solver → 运行计算
8. export_results → 导出结果

## 响应格式

- 每执行一个步骤，向用户报告进度和结果
- 遇到错误时，说明原因和正在采取的修复措施
- 使用 Markdown 格式化输出
"""


class SimulationAgent:
    """AI Agent：理解意图 → 规划步骤 → 调用工具 → 验证结果"""

    def __init__(self, mcp_manager: MCPClientManager, emit_callback):
        self.mcp = mcp_manager
        self.emit = emit_callback  # socketio.emit 的回调
        self.llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.conversation_history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.max_tool_rounds = 20  # 防止无限循环

    async def handle_message(self, user_message: str):
        """处理用户消息的主入口"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        try:
            await self._agent_loop()
        except Exception as e:
            logger.exception("Agent loop error")
            self.emit("bot_message", {
                "type": "text",
                "content": f"❌ Agent 处理异常: `{str(e)}`",
            })

    async def _agent_loop(self):
        """Agent 的 ReAct 循环：思考 → 行动 → 观察 → 重复"""
        for round_num in range(self.max_tool_rounds):
            # ── 调用 LLM ──
            response = await self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.conversation_history,
                tools=self.mcp.get_openai_tools() or None,
                tool_choice="auto",
                temperature=0.1,  # 仿真场景需要确定性输出
            )

            message = response.choices[0].message

            # ── 如果 LLM 没有调用工具，直接返回文本 ──
            if not message.tool_calls:
                assistant_text = message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_text,
                })
                self.emit("bot_message", {
                    "type": "text",
                    "content": assistant_text,
                })
                return

            # ── 处理工具调用 ──
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 向前端报告正在执行的步骤
                self.emit("simulation_log", {
                    "line": f"🔧 调用工具: {func_name}({json.dumps(func_args, ensure_ascii=False)})\n"
                })

                # 通过 MCP 调用工具
                result = await self.mcp.call_tool(func_name, func_args)

                # 解析结果并检查
                try:
                    result_data = json.loads(result)
                    success = result_data.get("success", True)
                except (json.JSONDecodeError, AttributeError):
                    result_data = {"raw": result}
                    success = True

                # 向前端报告结果
                status_icon = "✅" if success else "❌"
                self.emit("simulation_log", {
                    "line": f"  {status_icon} 结果: {result[:200]}\n"
                })

                # 将工具结果加入对话历史
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # 超过最大轮次
        self.emit("bot_message", {
            "type": "text",
            "content": "⚠️ Agent 执行步骤过多，已停止。请检查仿真配置。",
        })


# ══════════════════════════════════════════════════════════
#  Flask Web 应用
# ══════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = "fluent-agent-2026"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# 全局对象
mcp_manager = MCPClientManager()
agent: SimulationAgent | None = None
loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """在事件循环中运行协程"""
    if loop is None:
        raise RuntimeError("Event loop not initialized")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=600)  # 仿真可能需要较长时间


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/report_images/<path:filename>")
def serve_image(filename):
    return send_from_directory(
        os.path.join(WORK_DIR, "report_images"), filename
    )


@socketio.on("connect")
def handle_connect():
    socketio.emit("bot_message", {
        "type": "text",
        "content": (
            "👋 你好！我是 **PyFluent 仿真智能助手 (Agent 模式)**\n\n"
            "与传统脚本不同，我会：\n"
            "- 🧠 **理解你的意图**，自动规划仿真步骤\n"
            "- ✅ **验证每个参数**，防止无效设置\n"
            "- 🔧 **自动诊断错误**，尝试修复问题\n"
            "- 📊 **分步报告进度**，每步可视化\n\n"
            "试试说：\n"
            "- 「启动 Fluent 并读取 elbow.msh 网格」\n"
            "- 「这是一个内流传热问题，请推荐合适的湍流模型」\n"
            "- 「设置入口速度 5 m/s，温度 300K，出口压力 0 Pa」\n"
            "- 「运行 200 步迭代」\n"
        ),
    })


@socketio.on("user_message")
def handle_user_message(data):
    """接收用户消息，交给 Agent 处理"""
    msg = data.get("message", "").strip()
    if not msg:
        return

    global agent
    if agent is None:
        socketio.emit("bot_message", {
            "type": "text",
            "content": "⏳ Agent 正在初始化，请稍候...",
        })
        return

    socketio.emit("simulation_start", {})

    try:
        run_async(agent.handle_message(msg))
    except Exception as e:
        socketio.emit("bot_message", {
            "type": "text",
            "content": f"❌ 处理失败: `{str(e)}`",
        })
    finally:
        socketio.emit("simulation_end", {})


# ══════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════

def start_event_loop(lp: asyncio.AbstractEventLoop):
    """在独立线程中运行 asyncio 事件循环"""
    asyncio.set_event_loop(lp)
    lp.run_forever()


def emit_callback(event: str, data: dict):
    """socketio.emit 的包装，供 Agent 使用"""
    socketio.emit(event, data)


if __name__ == "__main__":
    import threading

    # 启动 asyncio 事件循环（独立线程）
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=start_event_loop, args=(loop,), daemon=True)
    loop_thread.start()

    # 初始化 MCP 连接
    async def init():
        global agent
        await mcp_manager.connect_all(MCP_SERVERS)
        agent = SimulationAgent(mcp_manager, emit_callback)
        logger.info(
            f"Agent 初始化完成，可用工具: "
            f"{list(mcp_manager.tool_registry.keys())}"
        )

    asyncio.run_coroutine_threadsafe(init(), loop).result(timeout=30)

    print("=" * 60)
    print("  PyFluent 仿真 Agent（MCP 架构）")
    print(f"  可用工具: {list(mcp_manager.tool_registry.keys())}")
    print("  访问地址: http://localhost:5000")
    print("=" * 60)

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
