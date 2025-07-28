from typing import Optional

from aiogram import Router, F, html
from aiogram.types import InlineQuery, \
    InlineQueryResultArticle, InputTextMessageContent, \
    InlineQueryResultCachedPhoto

from storage import get_links_by_id, get_images_by_id

router = Router()

@router.inline_query(F.query == "links")
async def show_user_links(inline_query: InlineQuery):
    # This function simply 