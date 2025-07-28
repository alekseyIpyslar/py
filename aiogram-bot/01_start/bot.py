import asyncio
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.enums import DiceEmoji
from aiogram.filters.command import Command

from config_reader import config

logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()
dp["started_at"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Hello!")

async def main():
    await dp.start_polling(bot)

@dp.message(Command("test1"))
async def cmd_test1(message: types.Message):
    await message.reply("Test 1")

async def cmd_test2(message: types.Message):
    await message.reply("Test 2")

@dp.message(Command("answer"))
async def cmd_answer(message: types.Message):
    await message.answer("It's a simple answer!")

@dp.message(Command("reply"))
async def cmd_reply(message: types.Message):
    await message.reply('It\'s a answer with "reply"!')

@dp.message(Command("dice"))
async def cmd_dice(message: types.Message):
    await message.answer_dice(emoji=DiceEmoji.DICE)

# @dp.message(Command("dice"))
# async def cmd_dice(message: types.Message, bot: Bot):
#     await bot.send_dice(-100123456789, emoji=DiceEmoji.DICE)

@dp.message(Command("add_to_list"))
async def cmd_add_to_list(message: types.Message, mylist: list[int]):
    mylist.append(7)
    await message.answer("Number 7 added")

@dp.message(Command("show_list"))
async def cmd_show_list(message: types.Message, mylist: list[int]):
    await message.answer(f"Your list is: {mylist}")

@dp.message(Command("info"))
async def cmd_info(message: types.Message, started_at: str):
    await message.answer(f"Bot started at: {started_at}")

async def main():
    dp.message.register(cmd_test2, Command("test2"))

    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, mylist=[1, 2, 3])

if __name__ == "__main__":
    asyncio.run(main())

