from enum import Enum


class QueryType(str, Enum):
    TEXT = 'Text'
    OCR = 'OCR'
    OBJECT = 'Object'
    AUDIO = 'Audio'
    TEMPORAL = 'Temporal'


__all__ = ["QueryType"]
