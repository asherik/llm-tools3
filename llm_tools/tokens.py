from dataclasses import dataclass
from typing import List, Dict
import logging
import tiktoken

from llm_tools.chat_message import (
    OpenAIChatMessage,
    prepare_messages,
    convert_message_to_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class TokenExpense:
    model_name: str
    n_input_tokens: int = 0
    n_output_tokens: int = 0

    @property
    def n_total_tokens(self) -> int:
        return self.n_input_tokens + self.n_output_tokens

    def __add__(self, other: "TokenExpense") -> "TokenExpense":
        if other.model_name != self.model_name:
            raise ValueError("Cannot add TokenExpense objects with different model names")
        return TokenExpense(
            model_name=self.model_name,
            n_input_tokens=self.n_input_tokens + other.n_input_tokens,
            n_output_tokens=self.n_output_tokens + other.n_output_tokens,
        )

    def price_per_1e6_input_tokens(self) -> int:
        return {
            "gpt-3.5-turbo": 2,
            "gpt-4": 30,
            "gpt-4-turbo": 10,
            "gpt-4o": 5,
            "gpt-4o-mini": 1,  # TODO не корректно, надо переводить на double (0.3)
        }[self.model_name]

    def price_per_1e6_output_tokens(self) -> int:
        return {
            "gpt-3.5-turbo": 2,
            "gpt-4": 60,
            "gpt-4-turbo": 30,
            "gpt-4o": 15,
            "gpt-4o-mini": 2,  # TODO не корректно, надо переводить на double (1.2)
        }[self.model_name]

    def get_price_multiplied_by_1e6(self) -> int:
        return (
                self.price_per_1e6_input_tokens() * self.n_input_tokens
                + self.price_per_1e6_output_tokens() * self.n_output_tokens
        )

    def get_price(self) -> float:
        return self.get_price_multiplied_by_1e6() / 1e6


@dataclass
class TokenExpenses:
    expenses: Dict[str, TokenExpense]

    def __init__(self):
        self.expenses = {}

    def add_expense(self, expense: TokenExpense):
        if expense.model_name in self.expenses:
            self.expenses[expense.model_name] += expense
        else:
            self.expenses[expense.model_name] = expense

    def __add__(self, other: "TokenExpenses") -> "TokenExpenses":
        result = TokenExpenses()
        for expense in self.expenses.values():
            result.add_expense(expense)
        for expense in other.expenses.values():
            result.add_expense(expense)
        return result

    def get_price_multiplied_by_1e6(self) -> int:
        return sum(expense.get_price_multiplied_by_1e6() for expense in self.expenses.values())

    def get_price(self) -> float:
        return self.get_price_multiplied_by_1e6() / 1e6


def count_tokens_from_input_messages(
        messages: List[OpenAIChatMessage],
        model_name: str,
) -> int:
    if not messages:
        return 0
    messages_typed = prepare_messages(messages)
    message_dicts = [convert_message_to_dict(x) for x in messages_typed]
    return num_tokens_from_messages(
        messages=message_dicts,
        model=model_name,
    )


def count_tokens_from_output_text(
        text: str,
        model_name: str,
) -> int:
    if not text:
        return 0
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))


def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        logger.debug("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo")
    elif "gpt-4" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming pt-4.")
        return num_tokens_from_messages(messages, model="gpt-4")
    elif "gpt-4o" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4o.")
        return num_tokens_from_messages(messages, model="gpt-4o")
    elif "gpt-4o-mini" in model:
        logger.debug("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4o-mini.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # Check if the value is a list
            if isinstance(value, list):
                for item in value:
                    # Check for the 'image_url' type within each dictionary in the list
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        num_tokens += 300 #openai берет такую цену при low качестве за каждую картинку, тут немного с запасом
                    else:
                        # Encode the value if it's not an image
                        num_tokens += len(encoding.encode(str(item)))
            else:
                # Handle non-list values
                num_tokens += len(encoding.encode(value))

            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

