import argparse
import numpy as np
import random
from PIL import Image
from tqdm import tqdm

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]

dirr = [0, 0, 1, -1]
dicc = [1, -1, 0, 0]
def random_walk(image_plane, x, y, ratio):
    height = image_plane.shape[0]
    width = image_plane.shape[1]
    steps = ratio * (height * width)
    active_pts = [(x, y)]
    hole_pts = [(x, y)]
    cnt = 1
    vis = [[0 for i in range(width)] for j in range(height)]
    while(cnt < steps):
        r = [i for i in range(4)]
        random.shuffle(r)
        ind = random.randint(0, len(active_pts) - 1)
        pt = active_pts[ind]
        j = 0
        x, y = pt[0], pt[1]
        while(True):
            if j == 4:
                active_pts.remove(active_pts[ind])
                break
            hori, vert = dirr[r[j]], dicc[r[j]]
            new_x = np.clip(x + hori, a_min=0, a_max=width - 1)
            new_y = np.clip(y + vert, a_min=0, a_max=height - 1)
            if(vis[new_x][new_y]):
                j += 1
                continue
            vis[new_x][new_y] = 1
            hole_pts.append((new_x, new_y))
            active_pts.append((new_x, new_y))
            cnt += 1
            break
    hole_pts = np.array(hole_pts)
    X = hole_pts[:, 0]
    Y = hole_pts[:, 1]
    image_plane[X, Y] = 0
    print(len(hole_pts), height * width)
    return image_plane
        
        



if __name__ == '__main__':
    import os

    RATIO = 0.1

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256) #shivam: changed 512 to 256 for celebhq
    parser.add_argument('--N', type=int, default=30000)
    parser.add_argument('--save_dir', type=str, default='masks/ratio' + str(RATIO * 100))
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # start, end = [int(j) for j in input().split()]
    start, end = 1, 100

    for i in tqdm(range(start,end+1)):
        canvas = np.ones((args.image_size, args.image_size)).astype("i")
        ini_x = random.randint(0, args.image_size - 1)
        ini_y = random.randint(0, args.image_size - 1)
        mask = random_walk(canvas, ini_x, ini_y, RATIO)
        # print("save:", i, np.sum(mask))

        img = Image.fromarray(mask * 255).convert('1')
        img.save('{:s}/{:05d}.jpg'.format(args.save_dir, i)) #shivam: changed 06d to 05d for celebhq
