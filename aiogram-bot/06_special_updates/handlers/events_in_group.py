from aiogram import Router, F
from aiogram.filters.command import Command
from aiogram.types import Message

router = Router()

# Generally speaking, you can attach a custom filter to the router with a check whether the
# caller's ID is in the admins set. Then all handlers in the router will be automatically
# called only for people from admins, this will shorten the code and get rid of unnecessary
# if, but for the sake of example we will do it through if-else, so that it is clearer

@router.message(Command("ban"), F.reply_to_message)
async def cmd_ban(message: Message, admins: set[int]):
    if message.from_user.id not in admins:
        await message.answer("You do not have permission to perform this command.")
    else:
        await message.chat.ban(
            user_id=message.reply_to_message.from_user.id
        )
        await message.answer("Intruder blocked")