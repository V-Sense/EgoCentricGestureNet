import argparse
import fnmatch
import os
import os.path
import torch

from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from model import ego_gesture_net


def read_sequence_to_tensor(dir_path, img_transforms):
    img_files = []
    if os.path.exists(dir_path):
        img_files = sorted(fnmatch.filter(os.listdir(dir_path), '*.png'))
        img_files = [os.path.join(dir_path, img_file) for img_file in img_files]
    num_imgs = len(img_files)
    img_tensor = torch.zeros([num_imgs, 3, 126, 224])
    for img_file, count in zip(img_files, range(num_imgs)):
        img = Image.open(img_file).convert("RGB")
        if img_transforms:
            img = img_transforms(img)
            img_tensor[count, :, :, :] = img
    return img_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", help="full path to the test directory")
    args = parser.parse_args()

    img_transform = transforms.Compose([transforms.Resize((126, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])

    md = ego_gesture_net.EgoGestureNet().cuda()
    md.load_state_dict(torch.load('./model/ego_gesture_net.pth'))
    base_dir = args.test_dir

    gestures = range(1, 11)
    for gesture_id in gestures:
        gesture_dir = base_dir + '{:02d}'.format(gesture_id) + '/'
        samples = os.listdir(gesture_dir)
        samples = [os.path.join(gesture_dir, sample) for sample in samples]
        for sample in samples:
            img_tensor = read_sequence_to_tensor(sample, img_transform)
            img_tensor = Variable(img_tensor.cuda())
            print('Actual Gesture :: ', gesture_id, 'Recognised Gesture :: ', md.recognise_gesture(img_tensor)[0])


if __name__ == '__main__':
    main()

