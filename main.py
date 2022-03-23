from argparse import ArgumentParser
from pathlib import Path
import rosbag
from tqdm.auto import tqdm
from cv_bridge import CvBridge
from PIL import Image
import numpy as np
import yaml


class Topics:
    _topics = {"aligned_info": "/camera/aligned_depth_to_color/camera_info",
              "depth": "/camera/aligned_depth_to_color/image_raw",
              "info": "/camera/color/camera_info",
              "rgb": "/camera/color/image_raw"}

    @classmethod
    def topics(cls):
        return cls._topics.values()
    @classmethod
    def aligned_info(cls):
        return cls._topics['aligned_info']
    @classmethod
    def aligned_depth(cls):
        return cls._topics['depth']
    @classmethod
    def info(cls):
        return cls._topics['info']
    @classmethod
    def rgb(cls):
        return cls._topics['rgb']


def create_dict_params(message):
    params = {"D": message.D,
              "K": message.K,
              "R": message.R,
              "P": message.P,
              'height': message.height,
              'width': message.width,
              "distortion_model": message.distortion_model}
    return params


def same_image(img, path, time):
    time = "{:.6f}".format(time)
    Image.fromarray(img).save(f"{path}/{time}.png")


def parse_ros_bag(bag_file, save_dir, redundancy):
    with rosbag.Bag(bag_file, 'r') as bag_to_read:
        rgb_gen = bag_to_read.read_messages(topics=Topics.rgb())
        depth_gen = bag_to_read.read_messages(topics=Topics.aligned_depth())
        depth_info_gen = bag_to_read.read_messages(topics=Topics.aligned_info())
        info_gen = bag_to_read.read_messages(topics=Topics.info())
        bridge = CvBridge()
        save_params = True
        rgb_dir = f"./{save_dir}/rgb"
        depth_dir = f"./{save_dir}/depth"
        Path(rgb_dir).mkdir(parents=True, exist_ok=True)
        Path(depth_dir).mkdir(parents=True, exist_ok=True)
        i = 0
        for rgb, depth, depth_info, info in tqdm(zip(rgb_gen, depth_gen, depth_info_gen, info_gen)):
            if i != 0:
                if i >= redundancy:
                    i = 0
                else:
                    i += 1
                continue
            else:
                i += 1

            assert depth_info.message.D == info.message.D
            assert depth_info.message.K == info.message.K
            assert depth_info.message.R == info.message.R
            assert depth_info.message.P == info.message.P

            img = bridge.imgmsg_to_cv2(rgb.message, desired_encoding='passthrough')
            same_image(img, rgb_dir, rgb.message.header.stamp.to_sec())

            depth_img = np.frombuffer(depth.message.data, np.uint16).reshape(480, 640)
            same_image(depth_img, depth_dir, depth.message.header.stamp.to_sec())

            if save_params:
                with open(f'./{save_dir}/camera_info.yaml', 'w') as params_file:
                    yaml.dump(create_dict_params(info.message), params_file)
                save_params = False


if __name__ == '__main__':
    parser = ArgumentParser(description='Process rosbag files')
    parser.add_argument('-b', help='rosbag file to process')
    parser.add_argument('-d', help='dataset name')
    parser.add_argument('-r', help='redundancy (int)')

    args = parser.parse_args()
    bag_file_path = Path(args.b)
    save_dir = Path(args.d)
    redundancy = int(args.r)

    parse_ros_bag(bag_file_path, save_dir, redundancy)
    print("Done!")
