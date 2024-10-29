import supervision as sv
from supervision.geometry.core import Position
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class YoloObjectDetection:

    def __init__(self, model):

        self.model = model
        self.frame = None
        self.detections = None
        self. color_codes = ["#FF0000"]  # Red

    def predict(self,  q_img):
        try:
            frame = q_img.get()

            color_plate = sv.ColorPalette.from_hex(self.color_codes)

            triangle_annotator = sv.TriangleAnnotator(color=color_plate, outline_thickness=1, base=30, height=30)

            label_annotator = sv.LabelAnnotator(text_padding=8, text_position=Position.CENTER, text_scale=1, text_thickness=2)

            result = self.model.track(source=frame, agnostic_nms=True, iou=0.5, imgsz=640, conf=0.75, persist=True, verbose=False, classes=0)

            result = result[0]

            detections = sv.Detections.from_ultralytics(result)

            if detections:

                person_detect = detections[detections.class_id == 0]
                detections.xyxy = sv.pad_boxes(xyxy=person_detect.xyxy, px=6)

                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                labels = [
                    f"{tracker_id} {self.model.names[class_id]}"
                    for box, mask, confidence, class_id, tracker_id, class_name
                    in detections
                ]

                frame = triangle_annotator.annotate(
                    scene=frame,
                    detections=detections
                )

                frame = label_annotator.annotate(
                    scene=frame, detections=detections, labels=labels)

                return frame
            else:
                return frame
        except Exception as er:
            print(er)
