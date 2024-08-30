from typing import Dict, Optional
from beanie import Document, Indexed
from pydantic import Field

# Định nghĩa mô hình Keyframe với chỉ mục cho các trường con trong tags
class Keyframe(Document):
    key: Indexed(int, unique=True) = Field(default=0) #type: ignore
    path: Optional[str] = None
    tags: Dict[str, float] = Field(default_factory=dict)

    class Settings:
        collection = "keyframes"
        indexes = [
            "key",  # Tạo chỉ mục cho trường 'key'
        ]
        # Tạo chỉ mục cho các tag trong tags
        # Mongodb chỉ hỗ trợ chỉ mục cho các trường con của một document
        # vì vậy chúng ta sẽ tạo một chỉ mục phức hợp cho các tag trong tags
        # Đây là cách tạo chỉ mục cho các tag trong tags
        # Trong thực tế, bạn có thể cần phải thực hiện thêm các bước nếu cần chỉ mục phức tạp hơn
        index_options = {"tags": {"keys": ["tags"], "name": "tags_index"}}
