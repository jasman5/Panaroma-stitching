import argparse
import cv2
from stitcher import stitch_images
from utils import read_images


def parse_args():
    p = argparse.ArgumentParser(description='Advanced panorama stitcher using OpenCV')
    p.add_argument('--input', nargs='+', required=True, help='input images in left-to-right order')
    p.add_argument('--output', required=True, help='output panorama file')
    p.add_argument('--focal', type=float, default=None, help='focal length in pixels. default: image width')
    p.add_argument('--no-cylinder', dest='cylinder', action='store_false', help='disable cylindrical projection')
    return p.parse_args()


def main():
    args = parse_args()
    imgs = read_images(args.input)
    focal = args.focal if args.focal is not None else imgs[0].shape[1]
    pano = stitch_images(imgs, focal=focal, auto_cyl=args.cylinder)
    cv2.imwrite(args.output, pano)
    print(f"Saved panorama to {args.output}")


if __name__ == '__main__':
    main()