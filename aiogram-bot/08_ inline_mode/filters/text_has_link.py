from typing import Union, Dict, Any

from aiogram.filters import BaseFilter
from aiogram.types import Message

class HasLinkFilter(BaseFilter):
    async def __call__(self, message: Message) -> Union[bool, Dict[str, Any]]:
        # If there are no entities at all, return None,
        # in this case consider it an empty list
        entities = message.entities or []

        # If there is at least one link, return it
        for entity in entities:
            if entity.type == "url":
                return {"link": entity.extract_from(message.text)}

        # If nothing os found, return None
        return False