# client.py
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional, List, Dict, Any, Tuple, Awaitable, Callable
from openai import AsyncOpenAI
from contextlib import AsyncExitStack
import json
import asyncio
import os
import httpx
import sys

from prompts import *
from search_mcp import logger

# =========================
# 配置 & 工具函数
# =========================

DEFAULT_API_BASE = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "aaa")
DEFAULT_API_MODEL = os.getenv("OPENAI_MODEL", "deepseek/deepseek-chat:free")

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # api | ollama | auto
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_OPENAI_BASE = f"{OLLAMA_HOST.rstrip('/')}/v1"
OLLAMA_POLL_INTERVAL = float(os.getenv("OLLAMA_POLL_INTERVAL", "5"))

def get_clear_json(text: str) -> Tuple[int, str]:
    if '```json' not in text:
        return 0, text
    return 1, text.split('```json')[1].split('```')[0]

# 实时事件回调类型
EventCallback = Optional[Callable[[dict], Awaitable[None]]]


# =========================
# Ollama 模型监视器
# =========================
class OllamaModelWatcher:
    def __init__(self, host: str = OLLAMA_HOST, poll_interval: float = OLLAMA_POLL_INTERVAL):
        self.host = host.rstrip('/')
        self.poll_interval = max(0.0, poll_interval)
        self._models: List[str] = []
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    @property
    def models(self) -> List[str]:
        return list(self._models)

    async def _fetch_models_once(self) -> List[str]:
        try:
            async with httpx.AsyncClient(base_url=self.host, timeout=5.0) as client:
                r = await client.get("/api/tags")
                r.raise_for_status()
                data = r.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return models
        except Exception as e:
            logger.warning(f"[Ollama] 无法获取本地模型列表：{e}")
            return []

    async def refresh_now(self):
        new_models = await self._fetch_models_once()
        added = sorted(set(new_models) - set(self._models))
        removed = sorted(set(self._models) - set(new_models))
        self._models = sorted(new_models)
        if added or removed:
            if added:
                logger.info(f"[Ollama] 新增本地模型：{added}")
            if removed:
                logger.info(f"[Ollama] 移除本地模型：{removed}")
            logger.info(f"[Ollama] 当前本地模型：{self._models}")
        return self._models, added, removed

    async def _poll_loop(self):
        await self.refresh_now()
        if self.poll_interval <= 0:
            return
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                await self.refresh_now()

    async def start(self):
        self._stop_event.clear()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def probe_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.host, timeout=3.0) as client:
                r = await client.get("/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    async def pick_default_model(self, preferred: Optional[str] = None) -> Optional[str]:
        models, _, _ = await self.refresh_now()
        if not models:
            return None
        if preferred and preferred in models:
            return preferred
        latest = [m for m in models if m.endswith(":latest")]
        return latest[0] if latest else models[0]


# =========================
# 统一 LLM 客户端
# =========================
class UnifiedLLM:
    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        model_name: str = DEFAULT_API_MODEL,
        ollama_host: str = OLLAMA_HOST,
        ollama_openai_base: str = OLLAMA_OPENAI_BASE,
        poll_interval: float = OLLAMA_POLL_INTERVAL,
    ):
        self.provider_mode = (provider or "auto").lower()
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.ollama_openai_base = ollama_openai_base
        self.poll_interval = poll_interval

        self.client: Optional[AsyncOpenAI] = None
        self.ollama: Optional[OllamaModelWatcher] = None
        self.active_provider: Optional[str] = None
        self.requested_model: Optional[str] = model_name

    async def init(self):
        mode = self.provider_mode
        if mode not in ("api", "ollama", "auto"):
            mode = "auto"

        if mode in ("ollama", "auto"):
            self.ollama = OllamaModelWatcher(self.ollama_host, self.poll_interval)
            alive = await self.ollama.probe_alive()
            if alive:
                await self.ollama.start()
                chosen = await self.ollama.pick_default_model(preferred=self.model_name if mode == "ollama" else None)
                if chosen:
                    self.model_name = chosen
                    self.client = AsyncOpenAI(
                        base_url=self.ollama_openai_base,
                        api_key=self.api_key or "ollama",
                    )
                    self.active_provider = "ollama"
                    logger.info(f"[LLM] 使用 Ollama（OpenAI 兼容）: base={self.ollama_openai_base}, model={self.model_name}")
                    return
                else:
                    logger.warning("[LLM] Ollama 可用，但本地没有已安装模型。回退到 API。")
            else:
                logger.info("[LLM] 未检测到运行中的 Ollama。")

        # 回退到 API
        self.client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        self.active_provider = "api"
        logger.info(f"[LLM] 使用 API: base={self.api_base}, model={self.model_name}")

    async def aclose(self):
        if self.ollama:
            await self.ollama.stop()

    async def maybe_refresh_ollama_model(self):
        if self.ollama:
            models, added, _ = await self.ollama.refresh_now()
            if added and self.model_name not in models:
                pick = await self.ollama.pick_default_model()
                if pick:
                    logger.info(f"[LLM] 发现新模型，自动切换为：{pick}")
                    self.model_name = pick

    async def create_chat_completion(self, messages: List[Dict[str, Any]]):
        await self.maybe_refresh_ollama_model()
        return await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )


# =========================
# MCP 客户端
# =========================
class MCPClient:
    def __init__(self, provider_override: Optional[str] = None, model_override: Optional[str] = None):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.provider_override = provider_override
        self.model_override = model_override
        self.llm = UnifiedLLM(
            provider=provider_override or DEFAULT_PROVIDER,
            api_base=DEFAULT_API_BASE,
            api_key=DEFAULT_API_KEY,
            model_name=model_override or DEFAULT_API_MODEL,
            ollama_host=OLLAMA_HOST,
            ollama_openai_base=OLLAMA_OPENAI_BASE,
            poll_interval=OLLAMA_POLL_INTERVAL,
        )

    async def connect_to_server(self, server_script_path: str):
        await self.llm.init()

        # 使用当前解释器启动 MCP 服务器（Windows 更稳）
        env = dict(os.environ)
        env_updates = {}
        if self.llm.api_base:
            env_updates.setdefault("OPENAI_BASE_URL", self.llm.api_base)
        if self.llm.api_key:
            env_updates.setdefault("OPENAI_API_KEY", self.llm.api_key)
        if self.llm.ollama_openai_base:
            env_updates.setdefault("OLLAMA_OPENAI_BASE", self.llm.ollama_openai_base)
        if self.provider_override:
            env_updates["LLM_PROVIDER"] = self.provider_override
        elif self.llm.active_provider:
            env_updates["LLM_PROVIDER"] = self.llm.active_provider
        if self.model_override:
            env_updates["OLLAMA_MODEL"] = self.model_override
            env_updates["OLLAMA_MODEL_PREFERENCE"] = self.model_override
            env_updates["OPENAI_MODEL"] = self.model_override
        elif self.llm.model_name:
            env_updates.setdefault("OPENAI_MODEL", self.llm.model_name)
        env.update({k: v for k, v in env_updates.items() if v is not None})

        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env=env
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"\nConnected to server with tools: {[tool.name for tool in tools]}")

        if self.llm.ollama:
            logger.info(f"[Ollama] 当前本地模型：{self.llm.ollama.models}")

    async def process_query(self, query: str) -> str:
        """原有非流式版本，保留"""
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        logger.info(f'available_tools:\n\n{available_tools}')

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + str(available_tools)},
            {"role": "user", "content": query}
        ]
        resp = await self.llm.create_chat_completion(messages)
        message = resp.choices[0].message
        logger.info(f'llm_output(tool call)：{message.content}')

        results = []
        while True:
            flag, json_text = get_clear_json(message.content)

            if flag == 0:
                resp = await self.llm.create_chat_completion([{"role": "user", "content": query}])
                return resp.choices[0].message.content

            json_text = json.loads(json_text)
            tool_name = json_text['name']
            tool_args = json_text['params']
            result = await self.session.call_tool(tool_name, tool_args)
            logger.info(f'tool name: \n{tool_name}\ntool call result: \n{result}')
            results.append(result.content[0].text)

            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": f'工具调用结果如下：{result}'})
            messages.append({"role": "user", "content": NEXT_STEP_PROMPT.format(query)})

            resp = await self.llm.create_chat_completion(messages)
            message = resp.choices[0].message
            logger.info(f'llm_output：\n{message.content}')

            if 'finish' in message.content:
                break

            messages.append({"role": "assistant", "content": message.content})

        messages.append({
            "role": "user",
            "content": FINISH_GENETATE.format('\n\n'.join(results), query)
        })
        resp = await self.llm.create_chat_completion(messages)
        return resp.choices[0].message.content

    async def process_query_stream(self, query: str, event_cb: EventCallback = None) -> str:
        """流式版本：在关键步骤通过 event_cb 推送事件，便于前端实时可视化"""
        async def emit(ev: dict):
            if event_cb:
                try:
                    await event_cb(ev)
                except Exception:
                    pass

        # 列出工具并告知前端
        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
        await emit({"type": "tools", "tools": [t["function"]["name"] for t in available_tools]})
        await emit({"type": "phase", "phase": "plan", "progress": 0.1})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + str(available_tools)},
            {"role": "user", "content": query}
        ]
        resp = await self.llm.create_chat_completion(messages)
        message = resp.choices[0].message
        await emit({"type":"llm_message","stage":"plan","text": message.content})

        results = []
        progress = 0.15

        while True:
            flag, json_text = get_clear_json(message.content)

            if flag == 0:
                await emit({"type":"phase","phase":"write","progress": max(progress,0.8)})
                resp = await self.llm.create_chat_completion([{"role": "user", "content": query}])
                final_text = resp.choices[0].message.content
                await emit({"type":"final_report","format":"markdown","content": final_text})
                await emit({"type":"phase","phase":"done","progress": 1.0})
                return final_text

            try:
                json_data = json.loads(json_text)
            except Exception as e:
                await emit({"type":"note","level":"warn","text": f"工具 JSON 解析失败：{e}"})
                resp = await self.llm.create_chat_completion([{"role":"user","content": query}])
                final_text = resp.choices[0].message.content
                await emit({"type":"final_report","format":"markdown","content": final_text})
                await emit({"type":"phase","phase":"done","progress": 1.0})
                return final_text

            tool_name = json_data.get('name')
            tool_args = json_data.get('params', {})
            await emit({"type":"tool_call","name":tool_name,"params":tool_args})

            result = await self.session.call_tool(tool_name, tool_args)
            tool_text = result.content[0].text if result.content else ""
            results.append(tool_text)
            await emit({"type":"tool_result","name":tool_name,"summary": tool_text[:500], "raw": tool_text})

            messages.append({"role": "assistant","content": message.content})
            messages.append({"role": "user","content": f'工具调用结果如下：{result}'})
            messages.append({"role": "user","content": NEXT_STEP_PROMPT.format(query)})

            progress = min(progress + 0.2, 0.75)
            await emit({"type":"progress","value":progress,"label":f"after {tool_name}"})
            await emit({"type":"phase","phase":"research","progress":progress})

            resp = await self.llm.create_chat_completion(messages)
            message = resp.choices[0].message
            await emit({"type":"llm_message","stage":"research","text": message.content})

            if 'finish' in message.content:
                await emit({"type":"phase","phase":"write","progress": max(progress,0.85)})
                break

            messages.append({"role": "assistant","content": message.content})

        messages.append({
            "role": "user",
            "content": FINISH_GENETATE.format('\n\n'.join(results), query)
        })
        resp = await self.llm.create_chat_completion(messages)
        final_text = resp.choices[0].message.content
        await emit({"type":"final_report","format":"markdown","content": final_text})
        await emit({"type":"phase","phase":"done","progress": 1.0})
        return final_text

    async def chat_loop(self):
        logger.info("\nMCP Client Started!")
        logger.info("Type your queries or 'quit' to exit.")
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(response)
            except Exception as e:
                logger.error(f"\nError: {str(e)}")

    async def close(self):
        await self.llm.aclose()
        await self.exit_stack.aclose()


async def main():
    client = MCPClient()
    try:
        await client.connect_to_server('./search_mcp.py')
        await client.chat_loop()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
