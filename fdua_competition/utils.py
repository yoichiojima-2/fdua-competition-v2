from pprint import pprint

from tenacity import RetryCallState


def log_retry(state: RetryCallState) -> None:
    pprint(f":( retrying attempt {state.attempt_number} after exception: {state.outcome.exception()}")
