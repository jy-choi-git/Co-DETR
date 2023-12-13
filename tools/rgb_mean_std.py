import cv2
import numpy as np
import os
from tqdm import tqdm

def get_rgb_mean_std(image_paths):
  """5만장 정도 되는 이미지의 r, g, b 채널별 평균과 표준편차를 구합니다.

  Args:
    image_paths: 이미지 경로의 리스트.

  Returns:
    r, g, b 채널별 평균과 표준편차의 리스트.
  """

  r_means = []
  g_means = []
  b_means = []
  r_stds = []
  g_stds = []
  b_stds = []

  for image_path in tqdm(image_paths):
    image = cv2.imread(image_path)
    rgb = cv2.split(image)

    r_mean = np.mean(rgb[0])
    g_mean = np.mean(rgb[1])
    b_mean = np.mean(rgb[2])

    r_std = np.std(rgb[0])
    g_std = np.std(rgb[1])
    b_std = np.std(rgb[2])

    r_means.append(r_mean)
    g_means.append(g_mean)
    b_means.append(b_mean)
    r_stds.append(r_std)
    g_stds.append(g_std)
    b_stds.append(b_std)

  return np.array(r_means), np.array(g_means), np.array(b_means), np.array(r_stds), np.array(g_stds), np.array(b_stds)

def get_all_files_in_folder_v2(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_list.append(os.path.join(root, filename))
    return file_list

if __name__ == "__main__":
    image_paths = get_all_files_in_folder_v2("/home/jinyoung/GitRepos/detection/data/fashionpedia/train")

    r_means, g_means, b_means, r_stds, g_stds, b_stds = get_rgb_mean_std(image_paths)

    print("r_means:", np.mean(r_means))
    print("g_means:", np.mean(g_means))
    print("b_means:", np.mean(b_means))
    print("r_stds:", np.mean(r_stds))
    print("g_stds:", np.mean(g_stds))
    print("b_stds:", np.mean(b_stds))