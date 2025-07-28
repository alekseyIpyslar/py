from aiogram import F, Router
from aiogram.filters import Command
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state
from aiogram.types import Message, ReplyKeyboardRemove

router = Router()

@router.message(Command(commands=["start"]))
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        text="Choose what you want to order: "
             "food (/food) or drinks (/drinks).",
        reply_markup=ReplyKeyboardRemove()
    )

# It's not hard to guess that the following two handlers can be
# safely combined into one, but for the sake of completeness, let's keep it that

# default_state - is the same as StateFilter(None)
@router.message(StateFilter(None), Command(commands=["cancel"]))
@router.message(default_state, F.text.lower() == "cancel")
async def cmd_cancel_no_state(message: Message, state: FSMContext):
    # We don't need to reset the state, just delete the data
    await state.set_data({})
    await message.answer(
        text="Nothing to change",
        reply_markup=ReplyKeyboardRemove()
    )

@router.message(Command(commands=["cancel"]))
@router.message(default_state, F.text.lower() == "cancel")
async def cmd_cancel_state(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        text="Action canceled",
        reply_markup=ReplyKeyboardRemove()
    )