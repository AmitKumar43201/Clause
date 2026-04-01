from typing import Any
from agent.agent import Agent
from agent.agentSchema import AgentEventType
from client.llm_client import LLMClient
from ui.tui import get_console, TUI
import click
import asyncio
import os
import sys

console = get_console()
class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI(console)

    async def run_single(self, message: str) -> str | None:
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)
        
    assistant_streaming = False
    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None
        final_response: str | None = None
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not self.assistant_streaming:
                    self.tui.begin_assistant()
                    self.assistant_streaming = True
                self.tui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
                if self.assistant_streaming:
                    self.tui.end_assistant()
                    self.assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown Eror")
                console.print(f"\n[error]Error: {error}[/error]")
        return final_response

@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None = None): 
    cli = CLI()
    if prompt:
        print(prompt)
        #messages = [{"role": "user", "content": prompt}]
        result = asyncio.run(cli.run_single(prompt))
        if result is None:
            sys.exit(1)
        print("done")


main()