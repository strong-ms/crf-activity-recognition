from typing import Optional, TypeVar, Generic

from pydantic import BaseModel

T = TypeVar('T')


class Result(Generic[T], BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[T] = None
