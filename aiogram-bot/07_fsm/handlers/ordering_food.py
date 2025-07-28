from aiogram import Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove

from keyboards.simple_row import make_row_keyboard
from tensorflow.python.framework.test_ops import None_

router = Router()

# These values will then be substituted into the final text, hence
# such a strange form of adjectives at first glance
available_food_names = ["Sushi", "Spaghetti", "Hachapuri"]
available_food_sizes = ["Small", "Medium", "Large"]

class OrderFood(StatesGroup):
    choosing_food_name = State()
    choosing_food_size = State()

@router.message(StateFilter(None), Command("food"))
async def cmd_food(message: Message, state: FSMContext):
    await message.answer(
        text="Select a dish:",
        reply_markup=make_row_keyboard(available_food_names)
    )
    # Set the user state to "chooses a name"
    await state.set_state(OrderFood.choosing_food_name)

# State of choosing a dish #

@router.message(OrderFood.choosing_food_name, F.text.in_(available_food_names))
async def food_chosen(message: Message, state: FSMContext):
    await state.update_data(chosen_food=message.text.lower())
    await message.answer(
        text="Thank you. Now please select your serving size:",
        reply_markup=make_row_keyboard(available_food_sizes)
    )
    await state.set_state(OrderFood.choosing_food_size)

# In general, no one prevents you from specifying states entirely as strings
# This can be useful if for some reason
# Your state names are generated at runtime (but why?)
@router.message(StateFilter("OrderFood:choosing_food_name"))
async def food_chosen_incorrectly(message: Message):
    await message.answer(
        text="I don't know such a dish.\n\n"
             "Please choose one of the names from the list below:",
        reply_markup=make_row_keyboard(available_food_names)
    )

# Portion size selection stage and summary display

@router.message(OrderFood.choosing_food_size, F.text.in_(available_food_sizes))
async def food_size_chosen(message: Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer(
        text=f"You have chosen {message.text.lower()} serving of {user_data['chosen_food']}.\n"
             f"Now try ordering drinks: /drinks",
        reply_markup=ReplyKeyboardRemove()
    )
    # Reset user state and saved data
    await state.clear()

@router.message(OrderFood.choosing_food_size)
async def food_size_chosen_incorrectly(message: Message):
    await message.answer(
        text="I don't know that serving size.\n\n"
             "Please select one of the options from the list below:",
        replt_markup=make_row_keyboard(available_food_sizes)
    )





















