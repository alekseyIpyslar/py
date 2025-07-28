from aiogram import F, Router, Bot
from aiogram.filters.chat_member_updated import \
    ChatMemberUpdatedFilter, IS_NOT_MEMBER, MEMBER, ADMINISTRATOR
from aiogram.types import ChatMemberUpdated

router = Router()
router.my_chat_member.filter(F.chat.type.in_({"group", "supergroup"}))

chats_variants = {
    "group": "group",
    "supergroup": "supergroup"
}

# Couldn't reproduce the case of adding a bot as Restricted, so there won't be an example with it

@router.my_chat_member(
    ChatMemberUpdatedFilter(
        member_status_changed=IS_NOT_MEMBER >> ADMINISTRATOR
    )
)
async def bot_added_as_admin(event: ChatMemberUpdated):
    # The simplest case: the bot is added as an admin
    # We can easily send a message
    await event.answer(
        text=f"Hello! Thanks for adding me in "
             f'{chats_variants[event.chat.type]} "{event.chat.title}"'
             f"as administrator. Chat ID: {event.chat.id}"
    )

@router.my_chat_member(
    ChatMemberUpdatedFilter(
        member_status_changed=IS_NOT_MEMBER >> MEMBER
    )
)
async def bot_added_as_member(event: ChatMemberUpdated, bot: Bot):
    # A more complicated option: the bot was added as a regular participant.
    # But it may not have the right to write messages, so we'll check in advance.
    chat_info = await bot.get_chat(event.chat.id)
    if chat_info.permissions.can_send_messages:
        await event.answer(
            text=f"Hello! Thanks for adding me in "
                 f'{chats_variants[event.chat.type]} "{event.chat.title}"'
                 f"as a regular member. Chat ID: {event.chat.id}"
        )
    else:
        print("Somehow we're gonna make sense of this situation")