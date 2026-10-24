#!/usr/bin/env python3
"""
Convert an image topic in a rosbag to an mp4 video.

Frames are written at a constant frame rate, and each bag frame is repeated until the next frame's timestamp, so the
video plays back in real time even though the detector publishes at a variable rate (e.g. pauses while turning).

Usage:
    python3 bag_to_mp4.py osod_demo.bag
    python3 bag_to_mp4.py osod_demo.bag -o demo.mp4 --topic /camera/color/image_with_boxes --fps 30 --speed 2
"""

import argparse
import os

import cv2
import numpy as np
import rosbag


def image_to_bgr(msg):
    """Convert a sensor_msgs/Image (rgb8, bgr8, or mono8) to a BGR numpy array."""
    channels = 1 if msg.encoding == 'mono8' else 3
    image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * channels]
    image = image.reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def main():
    parser = argparse.ArgumentParser(description='Convert a rosbag image topic to mp4')
    parser.add_argument('bag', help='Input .bag file')
    parser.add_argument('-o', '--output', help='Output .mp4 file (default: next to the bag)')
    parser.add_argument('--topic', default='/camera/color/image_with_boxes', help='Image topic to export')
    parser.add_argument('--fps', type=float, default=30.0, help='Output frame rate')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (2 = twice as fast)')
    parser.add_argument('--max-gap', type=float, default=2.0,
                        help='Gaps longer than this (s, bag time) are shortened to this, so long stalls do not freeze the video')
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.bag)[0] + '.mp4'

    writer = None
    first_time = None
    prev_time = None
    video_time = 0.0  # Bag time (s, after gap shortening) that the last written frame covers up to
    frames_written = 0
    num_msgs = 0
    last_frame = None

    with rosbag.Bag(args.bag) as bag:
        for _, msg, t in bag.read_messages(topics=[args.topic]):
            # Use the publish time in the header, falling back to the record time
            stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else t.to_sec()
            frame = image_to_bgr(msg)

            if writer is None:
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'avc1'), args.fps, (msg.width, msg.height))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (msg.width, msg.height))
                first_time = stamp
                prev_time = stamp

            # Advance the timeline, shortening long gaps, then fill the video with the previous frame up to now
            video_time += min(max(stamp - prev_time, 0.0), args.max_gap)
            prev_time = stamp
            target_frames = int(round(video_time / args.speed * args.fps))
            while last_frame is not None and frames_written < target_frames:
                writer.write(last_frame)
                frames_written += 1

            last_frame = frame
            num_msgs += 1

        if writer is None:
            print(f'No messages on {args.topic} in {args.bag}')
            return

        # Write the final frame once
        writer.write(last_frame)
        frames_written += 1
        writer.release()

    print(f'Wrote {output}: {num_msgs} bag frames -> {frames_written} video frames '
          f'({frames_written / args.fps:.1f} s at {args.fps:g} fps, bag span {prev_time - first_time:.1f} s)')


if __name__ == '__main__':
    main()
