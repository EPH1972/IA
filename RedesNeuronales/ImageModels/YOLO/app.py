import argparse
import supervision as sv

def main(video_file_path):
    
    print(f"Processing video: {video_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file_path")

    args = parser.parse_args()

    main(args.video_file_path)