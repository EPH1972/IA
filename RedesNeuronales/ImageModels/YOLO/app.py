import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
import torch

POLYGON = np.array([
    [287,509],
    [289,574],
    [2,585],
    [3,852],
    [179,899],
    [285,943],
    [1679,941],
    [1681,580],
    [551,499]
])
CLASSES = [2,3]

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()


polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
trace_annotator = sv.TraceAnnotator(trace_length=60)

def main(video_file_path):
    # Check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("CUDA is NOT available. Using CPU.")
        print("Tip: If you have a GPU, check 'nvidia-smi'. If it says 'ERR!', a reboot is required.")
        device = 'cpu'

    
    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)
    for i, frame in enumerate(frame_generator):
        print(f"Processing frame {i}")
        results = model(frame, device=device, verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)


        detections = detections[polygon_zone.trigger(detections=detections)]
        detections = detections[np.isin(detections.class_id, CLASSES)]
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id 
            in detections.tracker_id
        ]

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=POLYGON, color=sv.Color.RED, thickness=2)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_file_path")

    args = parser.parse_args()

    main(args.video_file_path)