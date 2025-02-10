from decimal import ROUND_HALF_UP, Decimal

from langchain_core.tools import tool


@tool
def divide_number(a: str, b: str) -> str:
    """
    divides two numbers.
    args:
        a: the dividend.
        b: the divisor.
    """
    return str(float(a) / float(b))


@tool
def round_number(number: str, decimals: str) -> str:
    """
    Rounds a number to a specified number of decimals using round half up.
    Args:
        number: the number to round.
        decimals: the number of decimals to round to.

    Example:
        round_number("1.25", "1") returns "1.3"
        "少数第二位を四捨五入" means turn 1.25 into 1.3
    """
    decimals_i = int(decimals)
    quantizer = Decimal("1") if decimals_i <= 0 else Decimal(f"0.{'0' * (decimals_i - 1)}1")
    number_d = Decimal(str(number))
    return str(number_d.quantize(quantizer, rounding=ROUND_HALF_UP))
