import asyncio
import logging
import re
from datetime import datetime

from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message, FSInputFile, URLInputFile, BufferedInputFile, LinkPreviewOptions
from aiogram.utils.formatting import as_list, as_marked_section, Bold, as_key_value, HashTag
from aiogram.utils.markdown import hide_link
from aiogram.utils.media_group import MediaGroupBuilder

from config_reader import config

bot = Bot(
    token=config.bot_token.get_secret_value(),
    default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    )
)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

@dp.message(Command("test"))
async def any_message(message: Message):
    await message.answer("Hello, <b>world</b>!", parse_mode=ParseMode.HTML)
    await message.answer(r"Hello, *world*\!", parse_mode=ParseMode.MARKDOWN_V2)
    # Or use this way
    # await message.answer("Hello, *world*\\!", parse_mode=ParseMode.MARKDOWN_V2)
    await message.answer("Message with <u>HTML-markup</u>")
    await message.answer("Message without <s>any markup</s!", parse_mode=None)

@dp.message(Command("hello"))
async def cmd_hello(message: Message):
    await message.answer(
        f"Hello, {html.bold(html.quote(message.from_user.full_name))}",
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("advanced_example"))
async def cmd_advanced_example(message: Message):
    content = as_list(
        as_marked_section(
            Bold("Success:"),
            "Test 1",
            "Test 3",
            "Test 4",
            marker="✅ ",
        ),
        as_marked_section(
            Bold("Failed:"),
            "Test 2",
            marker="❌ ",
        ),
        as_marked_section(
            Bold("Summary:"),
            as_key_value("Total", 4),
            as_key_value("Success", 3),
            as_key_value("Failed", 1),
            marker=" ",
        ),
        HashTag("#test"),
        sep="\n\n",
    )
    await message.answer(**content.as_kwargs())

@dp.message(Command("settimer"))
async def cmd_settimer(
        message: Message,
        command: CommandObject
):
    # If no arguments are passed
    # command.args will be None
    if command.args is None:
        await message.answer(
            "Error: no arguments passed"
        )
        return
    # Try to split the arguments into two parts by the first space encountered
    try:
        delay_time, text_to_send = command.args.split(" ", maxsplit=1)
    # If the result is less than two parts, a ValueError will be thrown
    except ValueError:
        await message.answer(
            "Error: incorrect command format. Example:\n"
            "/settimer <time> <message>"
        )
        return
    await message.answer(
        "Timer added!\n"
        f"Time: {delay_time}\n"
        f"Text: {text_to_send}"
    )

@dp.message(Command("gif"))
async def send_gif(message: Message):
    await message.answer_animation(
        animation="<file_id gifs>",
        caption="Today I:",
        show_caption_above_media=True
    )

@dp.message(Command("custom1", prefix="%"))
async def cmd_custom1(message: Message):
    await message.answer("I see a command!")

# Multiple prefixes can be specified............vv.......
@dp.message(Command("custom2", prefix="/!"))
async def cmd_custom2(message: Message):
    await message.answer("I see that one, too!")

@dp.message(Command("help"))
@dp.message(CommandStart(
    deep_link=True, magic=F.args == "help"
))
async def cmd_start_help(message: Message):
    await message.answer("This is a message with a reference")

@dp.message(CommandStart(
    deep_link=True,
    magic=F.args.regexp(re.compile(r'book_(\d+)'))
))
async def cmd_start_book(
        message: Message,
        command: CommandObject
):
    book_number = command.args.split("_")[1]
    await message.answer(f"Sending book №{book_number}")

@dp.message(Command("links"))
async def cmd_links(message: Message):
    # Two links that will go into the final post
    links_text = (
        "https://nnplus1.ru/news/2024/05/23/voyager-1-science-data"
        "\n"
        "https://t.me/telegram"
    )
    # Link disabled
    options_1 = LinkPreviewOptions(is_disabled=True)
    await message.answer(
        f"No link previews\n{links_text}",
        link_preview_options=options_1
    )

    # ---------------------- #

    # Small preview
    # To use prefer_small_media you must also specify an url
    options_2 = LinkPreviewOptions(
        url="https://nplus1.ru/news/2024/05/23/voyager-1-science-data",
        prefer_small_media=True
    )

    await message.answer(
        f"Small preview\n{links_text}",
        link_preview_options=options_2
    )

    # ---------------------- #

    # Large preview
    # To use prefer_large_media, you must also specify the url
    options_3 = LinkPreviewOptions(
        url="https://nplus1.ru/news/2024/05/23/voyager-1-science-data",
        prefer_large_media=True
    )
    await message.answer(
        f"Large preview\n{links_text}",
        link_preview_options=options_3
    )

    # ---------------------- #
    # Can be combined: small preview and positioning above the text
    options_4 = LinkPreviewOptions(
        url="https://nplus1.ru/news/2024/05/23/voyager-1-science-data",
        prefer_small_media=True,
        show_above_text=True
    )
    await message.answer(
        f"Small preview above the text\n{links_text}",
        link_preview_options=options_4
    )

    # ---------------------- #

    # You can select which link will be used for the preview
    options_5 = LinkPreviewOptions(
        url="https://t.me/telegram"
    )
    await message.answer(
        f"Preview not the first link\n{links_text}",
        link_preview_options=options_5
    )

@dp.message(Command("hidden_link"))
async def cmd_hidden_link(message: Message):
    await message.answer(
        f"{hide_link('https://telegra.ph/file/562a512448876923e28c3.png')}"
        f"Telegram documentation: *exists*\n"
        f"Users: *don't read the documentation*\n"
        f"Pear:"
    )

@dp.message(Command('images'))
async def cmd_images(message: Message):
    # This is where we put the file_id of the sent files so we can use them later on
    file_ids = []

    # To demonstrate BufferedInputFile, let's use the "classic" opening of a file via 'open()'.
    # But generally speaking, this method is best suited for sending bytes from RAM
    # after performing some manipulation, for example, editing via Pillow
    with open("buffer_emulation.jpg", "rb") as image_from_buffer:
        result = await message.answer_photo(
            BufferedInputFile(
                image_from_buffer.read(),
                filename="image from buffer.jpg"
            ),
            caption="Image from buffer"
        )
        file_ids.append(result.photo[-1].file_id)

    # Sending a file from the file system
    image_from_pc = FSInputFile("image_from_pc.jpg")
    result = await message.answer_photo(
        image_from_pc,
        caption="Image from file on pc"
    )
    file_ids.append(result.photo[-1].file_id)

    # Sending file by link
    image_from_url = URLInputFile("https://picsum.photos/seed/groosha/400/300")
    result = await message.answer_photo(
        image_from_url,
        caption="Image from url"
    )
    file_ids.append(result.photo[-1].file_id)
    await message.answer("Sent files:\n" + "\n".join(file_ids))

@dp.message(Command("album"))
async def cmd_album(message: Message):
    album_builder = MediaGroupBuilder(
        caption="A shared caption for a future album"
    )
    album_builder.add(
        type="photo",
        media=FSInputFile("image_from_pc.jpg")
        # caption="Media-specific caption"
    )
    # If we know the type right away, then instead of the general add
    # we can immediately call add_<type>
    album_builder.add_photo(
        # For links or file_id it's sufficient to specify the value
        media="https://picsum.photos/seed/groosha/400/300"
    )
    # album_builder.add_photo(
    #     media="<your file id>"
    # )
    await message.answer_media_group(
        # Don't forget to call build()
        media=album_builder.build()
    )

@dp.message(F.text)
async def extract_data(message: Message):
    data = {
        "url": "<N/A>",
        "email": "<N/A>",
        "code": "<N/A>"
    }
    entities = message.entities or []
    for item in entities:
        if item.type in data.keys():
            # Incorrect
            # data[item.type] = message.text[item.offset : item.offset+item.length]
            # Correct
            data[item.type] = item.extract_from(message.text)
    await message.reply(
        "Here's what I found:\n"
        f"URL: {html.quote(data['url'])}\n"
        f"E-mail: {html.quote(data['email'])}\n"
        f"Password: {html.quote(data['code'])}"
    )

# This handler is overridden by a higher handler, comment out that one to make this one work
@dp.message(F.text)
async def echo_with_time(message: Message):
    # Get the current time in the PC time zone
    time_now = datetime.now().strftime("%H:%M")
    # Create underlined text
    added_text = html.underline(f"Created at {time_now}")
    # Send a new message with the added text
    await message.answer(f"{message.html_text}\n\n{added_text}")

@dp.message(F.animation)
async def echo_gif(message: Message):
    await message.reply_animation(message.animation.file_id)

@dp.message(F.photo)
async def download_photo(message: Message, bot: Bot):
    await bot.download(
        message.photo[-1],
        destination=f"/tmp/{message.photo[-1].file_id}.jpg"
    )

@dp.message(F.sticker)
async def download_sticker(message: Message, bot: Bot):
    await bot.download(
        message.sticker,
        destination=f"/tmp/{message.sticker.file_id}.webp"
    )

@dp.message(F.new_chat_members)
async def somebody_added(message: Message):
    for user in message.new_chat_members:
        await message.reply(f"Hello, {user.full_name}")

async def main():
    # Run the bot and skip all accumulated incoming
    # Yes, this method can be called even if you're Polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())