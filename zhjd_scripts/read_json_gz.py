import gzip
import json
from pprint import pprint

path = "/home/robotlab/dataset/test_for_object_goal_navigation/hard/v5/test/content/8194nk5LbLH.json.gz"
path = "/home/robotlab/dataset/test_for_object_goal_navigation/hard/v5/test/content/2azQ1b91cZZ.json.gz"

# 1. 读取并解压 json.gz
with gzip.open(path, "rt", encoding="utf-8") as f:
    data = json.load(f)

# 2. 打印顶层结构
print("🔹 顶层键：", list(data.keys()))
print("🔹 episodes 数量：", len(data["episodes"]))

# 3. 打印第一个 episode 的主要字段
ep0 = data["episodes"][0]
print("\n=== 第一个 episode 的主要字段 ===")
for k in ep0.keys():
    if isinstance(ep0[k], (dict, list)):
        print(f"{k}: {type(ep0[k]).__name__}, 长度 = {len(ep0[k]) if hasattr(ep0[k], '__len__') else '-'}")
    else:
        print(f"{k}: {ep0[k]}")

# 4. 展开打印部分字段内容
print("\n=== 示例内容（局部） ===")
print("scene_id:", ep0["scene_id"])
print("start_position:", ep0["start_position"])
print("start_rotation:", ep0["start_rotation"])
print("object_category:", ep0["object_category"])
print("goals: ", ep0["goals"])
# pprint(ep0["goals"][:1])
print("shortest_paths[0]所有动作: ", ep0["shortest_paths"][0])
# print("shortest_paths[0]前5步动作: ")
# pprint(ep0["shortest_paths"][0][:5])
