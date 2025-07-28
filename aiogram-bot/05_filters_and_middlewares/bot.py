import asyncio

from aiogram import Bot, Dispatcher

from config_reader import config
from handlers import group_games, checkin, usernames
from middlewares.weekend import WeekendCallbackMiddleware


async def main():
    bot = Bot(token=config.bot_token.get_secret_value())
    dp = Dispatcher()

    dp.include_router(group_games.router)
    dp.include_router(checkin.router)
    dp.include_router(usernames.router)

    dp.callback_query.outer_middleware(WeekendCallbackMiddleware())

    # Launch the bot and skip all accumulated incoming
    # Yes, this method can be called even if you have polling
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())