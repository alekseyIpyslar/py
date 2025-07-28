from datetime import datetime
from typing import Callable, Dict, Any, Awaitable

from aiogram import BaseMiddleware
from aiogram.types import Message, CallbackQuery


def _is_weekend() -> bool:
    # 5 - saturday, 6 - sunday
    return datetime.utcnow().weekday() in (5, 6)

# This will be inner-middleware on messages
class WeekendMessageMiddleware(BaseMiddleware):
    async def __call__(
            self,
            handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
            event: Message,
            data: Dict[str, Any]
    ) -> Any:
        # If today is neither Saturday nor Sunday, then continue processing.
        if not _is_weekend():
            return await handler(event, data)
        # Otherwise, simply return None and processing stops

# This will be outer-middleware for any callbacks
class WeekendCallbackMiddleware(BaseMiddleware):
    async def __call__(
            self,
            handler: Callable[[CallbackQuery, Dict[str, Any]], Awaitable[Any]],
            event: CallbackQuery,
            data: Dict[str, Any]
    ) -> Any:
        # If today is not Saturday or Sunday, then continue processing.
        if not _is_weekend():
            return await handler(event, data)
        # Otherwise, respond to the callback ourselves and stop further processing
        await event.answer(
            "The bot doesn't work on weekends!",
            show_alert=True
        )
        return