from typing import Union, Dict, Any

from aiogram.filters import BaseFilter
from aiogram.types import Message


class HasUsernamesFilter(BaseFilter):
    async def __call__(self, message: Message) -> Union[bool, Dict[str, Any]]:
        # If there are no entities at all, None will be returned,
        # in this case we assume that this is an empty list
        entities = message.entities or []

        # We check any usernames and extract them from the text using the extract_from() method.
        # For more details, see the chapter on working with messages
        found_usernames = [
            item.extract_from(message.text) for item in entities
            if item.type == "mention"
        ]

        # If there are usernames, we "push" them into the handler
        # by name "usernames"
        if len(found_usernames) > 0:
            return {"usernames": found_usernames}
        # If no usernames were found, we return False
        return False