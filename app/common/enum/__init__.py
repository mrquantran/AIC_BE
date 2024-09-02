from enum import Enum


class QueryType(str, Enum):
    TEXT = 'Text'
    OCR = 'OCR'
    OBJECT = 'Object'
    TEMPORAL = 'Temporal'


__all__ = ["QueryType"]
