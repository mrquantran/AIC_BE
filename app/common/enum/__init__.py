from enum import Enum


class EQueryType(Enum):
    TEXT = 'Text'
    OBJECT = 'Object'

QueryType = Enum("EQueryType", ["Text", "Object"])


__all__ = ["QueryType"]
