import asyncio

from aiogram import Bot, Dispatcher

from config_reader import config
from handlers import questions, different_types

# Start bot
async def main():
    bot = Bot(token=config.bot_token.get_secret_value())
    dp = Dispatcher()

    # dp.include_router(questions.router, different_types.router)

    # Alternative option for registering routers one per line
    dp.include_router(questions.router)
    dp.include_router(different_types.router)

    # Launch the bot and skip all accumulated incoming
    # Yes, this method can be called even if you have polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())