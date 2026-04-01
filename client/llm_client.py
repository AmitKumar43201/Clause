import asyncio
from typing import Any, AsyncGenerator
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from .schema import StreamEvent, TextDelta, TokenUsage, StreamEventType

class LLMClient:
    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._max_retry : int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key='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',
                base_url='https://openrouter.ai/api/v1',
                default_headers={
                    "HTTP-Referer": "https://your-site-url.example",
                    "X-Title": "my-app"
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            
    async def chat_completion( self, messages: list[dict[str, Any]], stream: bool = True) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            "model": 'openai/gpt-oss-120b:free',
            "messages": messages,
            "stream": stream 
        }
        for attempt in range(self._max_retry + 1):
            try:
                if stream:
                    async for event in  self._stream_response(client=client, kwargs=kwargs):
                        yield event
                    return
                else:
                    res = await self._non_stream_response(client=client, kwargs=kwargs)
                    yield res
                    return
            except RateLimitError as e:
                if attempt<=self._max_retry:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Rate limit exceeded, error: {e}"
                    )
                    return
            except APIConnectionError as e:
                if attempt<=self._max_retry:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"Connection, error: {e}"
                    )
                    return
            except APIError as e:
                    yield StreamEvent(
                        type=StreamEventType.ERROR,
                        error=f"APIError, error: {e}"
                    )
                    return

    async def _stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]):
        response = await client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None

        async for chunk in response:
            
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(content=delta.content)
                )
                
            if choice.finish_reason:
                finish_reason= choice.finish_reason
            
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens,
                )
                     
        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage
        )

    async def _non_stream_response(self, client: AsyncOpenAI, kwargs: dict[str, Any]) -> StreamEvent:
        
        response = await client.chat.completions.create(**kwargs)
        #print(response)
        token_usage = response.usage
        choice = response.choices[0]
        message = choice.message
        #print(message)
        
        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)
            
        usage = None
        if token_usage:
            usage = TokenUsage(
                prompt_tokens= token_usage.prompt_tokens,
                completion_tokens= token_usage.completion_tokens,
                total_tokens= token_usage.total_tokens,
                cached_tokens= token_usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type= StreamEventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            usage= usage    
        )
    