from enum import Enum


class QueryType(str, Enum):
    TEXT = 'Text'
    OCR = 'OCR'
    OBJECT = 'Object'


__all__ = ["QueryType"]
