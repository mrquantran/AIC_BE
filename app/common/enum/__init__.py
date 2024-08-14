from enum import Enum


class QueryType(str, Enum):
    TEXT = 'Text'
    OBJECT = 'Object'


__all__ = ["QueryType"]
