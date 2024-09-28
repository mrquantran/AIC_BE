import json
import os
# Step 1: Đọc file JSON
with open(
    os.path.join(
        os.path.dirname(__file__), "merged_inference_results_numbered_output.json"
    )
) as f:
    data = json.load(f)

# Step 2: Tạo dictionary để lưu kết quả
result = {}

# Step 3: Duyệt qua từng index trong file JSON
for index, info in data.items():
    for tag, _ in info["tag_prob"]:
        # Step 4: Thêm index vào danh sách tương ứng với tag
        if tag not in result:
            result[tag] = []
        result[tag].append(int(index))

# Step 5: Ghi file JSON mới
with open(os.path.join(os.path.dirname(__file__), "tag_to_indices.json"), "w") as f:
    json.dump(result, f, indent=4)

print("Chuyển đổi hoàn tất!")
