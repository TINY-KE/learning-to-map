import gzip
import json
import os
import habitat
from habitat.config.default import get_config
import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_episodes_from_json_gz(file_path):
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    return data['episodes']

def convert_quaternion_to_rotation(quat_xyzw):
    # Habitat 使用 wxyz，但 episode 中一般是 xyzw
    x, y, z, w = quat_xyzw
    return [w, x, y, z]

def save_image(array, path):
    Image.fromarray(array).save(path)

def main():
    json_gz_path = "/home/robotlab/dataset/Test_Episodes/easy/v3/test/content/2azQ1b91cZZ.json.gz"  # 修改为实际路径
    scene_dataset_path = "data/scene_datasets/mp3d/"     # 修改为 MP3D 数据路径
    output_dir = "output_episodes"
    os.makedirs(output_dir, exist_ok=True)

    # Load episodes
    episodes = load_episodes_from_json_gz(json_gz_path)

    # Base config
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = ""  # We'll set it dynamically
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.WIDTH = 640
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 640
    config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 480
    config.SIMULATOR.SEMANTIC_SENSOR.MAP_HEIGHT = 480
    config.SIMULATOR.SEMANTIC_SENSOR.MAP_WIDTH = 640
    config.DATASET.SCENES_DIR = scene_dataset_path
    config.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)

    for i, ep in enumerate(episodes[:5]):  # 只处理前5个 episode 做示例
        scene_id = ep['scene_id']
        abs_scene_path = os.path.join(scene_dataset_path, scene_id)
        if not os.path.exists(abs_scene_path):
            print(f"❌ Scene not found: {abs_scene_path}")
            continue

        print(f"▶️ Episode {i}, Scene: {scene_id}")

        # 设置当前场景
        sim.reconfigure(config.SIMULATOR.clone())
        sim.config.SCENE = abs_scene_path
        sim.reconfigure(sim.config)

        # 设置起点与朝向
        sim.reset()
        start_pos = ep['start_position']
        start_rot_quat = convert_quaternion_to_rotation(ep['start_rotation'])

        sim.agents[0].scene_node.translation = np.array(start_pos)
        sim.agents[0].scene_node.rotation = habitat_sim.utils.quat_from_coeffs(start_rot_quat)

        # 获取传感器观测
        obs = sim.get_sensor_observations()

        rgb = obs['rgb']
        depth = obs['depth']
        semantic = obs['semantic']

        # 保存观测数据
        ep_prefix = os.path.join(output_dir, f"ep_{i}")
        os.makedirs(ep_prefix, exist_ok=True)

        save_image(rgb, os.path.join(ep_prefix, "rgb.png"))
        save_image((depth * 255).astype(np.uint8), os.path.join(ep_prefix, "depth.png"))
        save_image(semantic.astype(np.uint8), os.path.join(ep_prefix, "semantic.png"))

        # 保存位姿
        pose = sim.agents[0].get_state()
        with open(os.path.join(ep_prefix, "pose.txt"), "w") as f:
            f.write(f"position: {pose.position.tolist()}\n")
            f.write(f"rotation: {pose.rotation.tolist()}\n")

    sim.close()
    print("✅ 完成！")

if __name__ == "__main__":
    main()