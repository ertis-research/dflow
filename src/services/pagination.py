import math
from typing import List, TypeVar

from src.schemas import PaginatedMetaData, PaginatedResult

T = TypeVar("T")


def paginate(items: List[T], page: int, page_size: int) -> PaginatedResult[T]:
    """Slices `items` into the requested page. An out-of-range page is clamped to the last one."""
    total_items = len(items)
    total_pages = math.ceil(total_items / page_size) if page_size > 0 else 0

    current_page = max(0, min(page, total_pages - 1)) if total_pages > 0 else 0
    start = current_page * page_size
    end = start + page_size

    metadata = PaginatedMetaData(
        total_items=total_items,
        total_pages=total_pages,
        current_page=current_page,
        page_size=page_size,
        has_next=current_page < total_pages - 1,
        has_prev=current_page > 0,
    )
    return PaginatedResult(results=items[start:end], pagination=metadata)
