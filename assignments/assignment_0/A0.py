from typing import List, Union, Dict


def get_unique_elements(
    lst: List[Union[str, int]], n: int = 1
) -> List[Union[str, int]]:
    """Given a list of elements returns those that repeat at least n times. The
    output list should contain all unique elements and they should be returned
    in the same order as they first appear in the input list.

    Args:
        lst: Input list
        n (optional): Minimum number of times an element should be repeated to
            be returned. Defaults to 1.

    Returns:
        List of unique items
    """
    count: Dict[Union[str, int], int] = dict()
    for item in lst:
        count[item] = count.get(item, 0) + 1
    added: List[Union[str, int]] = []
    result: List[Union[str, int]] = []
    for item in lst:
        if item not in added and count[item] >= n:
            result.append(item)
            added.append(item)
    return result

