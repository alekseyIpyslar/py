import asyncio
import logging
from contextlib import suppress
from random import randint
from typing import Optional

from aiogram import Bot, Dispatcher, types, F
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

from config_reader import config

bot = Bot(token=config.bot_token.get_secret_value())
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

user_data = {}

@dp.message(Command("start"))
async def start(message: types.Message):
    kp = [
        [
            types.KeyboardButton(text="С пюрешкой"),
            types.KeyboardButton(text="Без пюрешки"),
        ],
    ]
    keyboard = types.ReplyKeyboardMarkup(
        keyboard=kp,
        resize_keyboard=True,
        input_field_placeholder="Select a submission method"
    )
    await message.answer("Как подавать котлеты?", reply_markup=keyboard)

@dp.message(F.text.lower() == "с пюрешкой")
async def with_puree(message: types.Message):
    await message.reply("Good choice!", reply_markup=types.ReplyKeyboardRemove())

@dp.message(F.text.lower() == "без пюрешки")
async def without_puree(message: types.Message):
    await message.reply("So tasteless!")

@dp.message(Command("reply_builder"))
async def reply_builder(message: types.Message):
    builder = ReplyKeyboardBuilder()
    for i in range(1, 17):
        builder.add(types.KeyboardButton(text=str(i)))
    builder.adjust(4)
    await message.answer(
        "Select a number:",
        reply_markup=builder.as_markup(resize_keyboard=True),
    )

@dp.message(Command("special_buttons"))
async def cmd_special_buttons(message: types.Message):
    builder = ReplyKeyboardBuilder()
    # The row method allows you to explicitly form a row of one or more buttons.
    # For example, the first row will consist of two buttons...
    builder.row(
        types.KeyboardButton(text="Request geolocation", request_location=True),
        types.KeyboardButton(text="Request contact", request_contact=True)
    )
    # ... second from one ...
    builder.row(types.KeyboardButton(
        text="Create a quiz",
        request_poll=types.KeyboardButtonPollType(type="quiz")
    ))
    # ... and the third one again from two
    builder.row(
        types.KeyboardButton(
            text="Select premium user",
            request_user=types.KeyboardButtonRequestUser(
                request_id=1,
                user_is_premium=True
            )
        ),
        types.KeyboardButton(
            text="Select a supergroup with forums",
            request_chat=types.KeyboardButtonRequestChat(
                request_id=2,
                chat_is_channel=False,
                chat_is_forum=True
            )
        )
    )
    await message.answer(
        "Select an action:",
        reply_markup=builder.as_markup(resizew_keyboard=True),
    )

@dp.message(F.user_shared)
async def on_user_shared(message: types.Message):
    print(
        f"Request {message.user_shared.request_id}. "
        f"User ID: {message.user_shared.user_id}"
    )

@dp.message(F.chat_shared)
async def on_chat_shared(message: types.Message):
    print(
        f"Request {message.chat_shared.request_id}. "
        f"Chat ID: {message.chat_shared.chat_id}"
    )

@dp.message(Command("inline_url"))
async def cmd_inline_url(message: types.Message, bot: Bot):
    builder = InlineKeyboardBuilder()
    builder.row(types.InlineKeyboardButton(
        text="GitHub", url="https://github.com"
    ))
    builder.row(types.InlineKeyboardButton(
        text="Off. channel Telegram",
        url="tg://resolve?domain=telegram"
    ))

    # To be able to show the ID button,
    # The user must have the has_private_forwards flag False
    user_id = 1234567890
    chat_info = await bot.get_chat(user_id)
    if not chat_info.has_private_forwards:
        builder.row(types.InlineKeyboardButton(
            text="Some user",
            url=f"tg://user?id={user_id}"
        ))
    await message.answer(
        'Select link',
        reply_markup=builder.as_markup(),
    )

@dp.message(Command("random"))
async def cmd_random(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="Click me",
        callback_data="random_value"
    ))
    await message.answer(
        "Click the button to have the bot send a number from 1 to 10",
        reply_markup=builder.as_markup()
    )

@dp.callback_query(F.data == "random_value")
async def send_random_value(callback: types.CallbackQuery):
    await callback.message.answer(str(randint(1, 10)))
    await callback.answer(
        text="Thank you for using the bot!",
        show_alert=True
    )
    # or just await call.answer()

# ----------
# This is variant without fabric.

def get_keyboard():
    buttons = [
        [
            types.InlineKeyboardButton(text="-1", callback_data="num_decr"),
            types.InlineKeyboardButton(text="+1", callback_data="num_incr")
        ],
        [types.InlineKeyboardButton(text="Approve", callback_data="num_finish")]
    ]
    keyboard = types.InlineKeyboardMarkup(inline_keyboard=buttons)
    return keyboard

async def update_num_text(message: types.Message, new_value: int):
    with suppress(TelegramBadRequest):
        await message.edit_text(
            f"Specify number: {new_value}",
            reply_markup=get_keyboard()
        )

@dp.message(Command("numbers"))
async def cmd_numbers(message: types.Message):
    user_data[message.from_user.id] = 0
    await message.answer("Specify number: 0", reply_markup=get_keyboard())

@dp.callback_query(F.data.startswith("num_"))
async def callbacks_num(callback: types.CallbackQuery):
    user_value = user_data.get(callback.from_user.id, 0)
    action = callback.data.split("_")[1]

    if action == "incr":
        user_data[callback.from_user.id] = user_value + 1
        await update_num_text(callback.message, user_value + 1)
    elif action == "decr":
        user_data[callback.from_user.id] = user_value - 1
        await update_num_text(callback.message, user_value - 1)
    elif action == "finish":
        await callback.message.edit_text(f"Total: {user_value}")

    await callback.answer()

# ----------
# This is variant with callback fabric

class NumbersCallbackFactory(CallbackData, prefix="fabnum"):
    action: str
    value: Optional[int] = None

def get_keyboard_fab():
    builder = InlineKeyboardBuilder()
    builder.button(text="-2", callback_data=NumbersCallbackFactory(action="change", value=-2))
    builder.button(text="-1", callback_data=NumbersCallbackFactory(action="change", value=-1))
    builder.button(text="+1", callback_data=NumbersCallbackFactory(action="change", value=1))
    builder.button(text="+2", callback_data=NumbersCallbackFactory(action="change", value=2))
    builder.button(text="Approve", callback_data=NumbersCallbackFactory(action="finish"))
    builder.adjust(4)
    return builder.as_markup()

async def update_num_text_fab(message: types.Message, new_value: int):
    with suppress(TelegramBadRequest):
        await message.edit_text(
            f"Specify number: {new_value}",
            reply_markup=get_keyboard_fab()
        )

@dp.message(Command("numbers_fab"))
async def cmd_numbers_fab(message: types.Message):
    user_data[message.from_user.id] = 0
    await message.answer("Specify number: 0", reply_markup=get_keyboard_fab())

# Pressing one of the buttons: -2, -1, +1, +2
@dp.callback_query(NumbersCallbackFactory.filter(F.action == "change"))
async def callbacks_num_change_fab(callback: types.CallbackQuery, callback_data: NumbersCallbackFactory):
    # Current value
    user_value = user_data.get(callback.from_user.id, 0)

    user_data[callback.from_user.id] = user_value + callback_data.value
    await update_num_text_fab(callback.message, user_value + callback_data.value)
    await callback.answer()

# Pressing the button "approve"
@dp.callback_query(NumbersCallbackFactory.filter(F.action == "finish"))
async def callbacks_num_finish_fab(callback: types.CallbackQuery):
    # Current value
    user_value = user_data.get(callback.from_user.id, 0)

    await callback.message.edit_text(f"Total: {user_value}")
    await callback.answer()

# Launch the bot
async def main():
    # Launch the bot and skip all accumulated incoming
    # Yes, this method can be called even if you have polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())