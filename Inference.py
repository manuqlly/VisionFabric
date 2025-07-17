import cv2
import numpy as np
import csv
from sort import Sort
from collections import defaultdict
from datetime import datetime

# Paths
CFG_PATH = 'custom-tiny.cfg'
WEIGHTS_PATH = 'Your Model.weights'
NAMES_PATH = 'obj.names'
VIDEO_INPUT = 'input_video'
VIDEO_OUTPUT = 'output.avi'

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4

def load_yolo_model(cfg_path, weights_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers, classes

def detect_towels(frame, net, output_layers, conf_threshold=0.5):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences = [], []
    for out in outputs:
        for detection in out:
            confidence = detection[4]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, NMS_THRESHOLD)
    final_dets = [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []
    return final_dets

def count_crossing_objects(tracker_outputs, roi_y1, roi_y2, counted_ids, object_states):
    count = 0
    for trk in tracker_outputs:
        x1, y1, x2, y2, obj_id = map(int, trk)
        center_y = int((y1 + y2) / 2)

        if obj_id not in object_states:
            object_states[obj_id] = center_y
            continue

        prev_y = object_states[obj_id]

        # Looser condition: crossing into the band from either direction
        crossed = (
            (prev_y < roi_y1 and center_y >= roi_y1) or
            (prev_y > roi_y2 and center_y <= roi_y2)
        )

        if obj_id not in counted_ids and crossed:
            counted_ids.add(obj_id)
            count += 1
            print(f"ID {obj_id} counted | Prev Y: {prev_y} → Curr Y: {center_y}")

        object_states[obj_id] = center_y
    return count

def draw_roi_and_boxes(frame, tracker_outputs, roi_y1, roi_y2, counted_ids):
    for trk in tracker_outputs:
        x1, y1, x2, y2, obj_id = map(int, trk)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Red dot for center
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"Y:{center_y}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Box color logic
        color = (0, 255, 0) if obj_id in counted_ids else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw ROI band
    cv2.rectangle(frame, (0, roi_y1), (frame.shape[1], roi_y2), (0, 255, 255), 2)
    cv2.putText(frame, "ROI Zone", (10, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_dashboard_panel(video_frame, towel_count, frame_num, logo_img):
    video_h, video_w = video_frame.shape[:2]
    panel_width = 250
    panel_height = video_h
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255  # white bg

    # Add logo at the top
    logo_height = 60
    top_padding = 10
    if logo_img is not None:
        logo_resized = cv2.resize(logo_img, (panel_width, logo_height))
        panel[top_padding:top_padding+logo_height, :] = logo_resized

    now = datetime.now()
    lines = [
        "LIVE Stats:",
        f"Date: {now.strftime('%Y-%m-%d')}",
        f"Time: {now.strftime('%H:%M:%S')}",
        "",
        "Current Details:",
        "Fabric name: xyz",
        "Batch no: 1234",
        "",
        "Current Stats:",
        f"Fabric Count: {towel_count}"
    ]

    bold_headers = {"LIVE Stats:", "Current Details:", "Current Stats:"}

    y_offset = top_padding + logo_height + 20  # Start below logo
    for line in lines:
        if line in bold_headers:
            cv2.putText(panel, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Bold header
            y_offset += 20
            cv2.line(panel, (10, y_offset), (panel_width - 10, y_offset), (0, 0, 0), 1)
            y_offset += 15
        elif line.strip() == "":
            y_offset += 10  # Blank line spacing
        else:
            cv2.putText(panel, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 2)  # Make text bold by using thickness=2
            y_offset += 22


    combined = np.hstack((video_frame, panel))
    return combined


def main():
    net, output_layers, _ = load_yolo_model(CFG_PATH, WEIGHTS_PATH, NAMES_PATH)
    cap = cv2.VideoCapture(VIDEO_INPUT)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width, height = original_height, original_width  # After 90 deg rotation

    # ROI band settings
    roi_band_height = 20
    roi_center = height - 300
    roi_y1 = roi_center - roi_band_height // 2
    roi_y2 = roi_center + roi_band_height // 2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, 30, (width + 250, height))


    tracker = Sort()
    # Load company logo (with white background, already prepared image)
    logo_img = cv2.imread("LOGO.png")

    towel_count = 0
    counted_ids = set()
    object_states = dict()

    with open("towel_count_log.csv", "w", newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["Frame", "Towel ID"])

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #comment it if you dont want to rotate the video
            frame_num += 1

            detections = detect_towels(frame, net, output_layers, CONF_THRESHOLD)
            dets_np = np.array([[x, y, x + w, y + h, 1.0] for (x, y, w, h) in detections])

            tracker_outputs = tracker.update(dets_np if dets_np.size > 0 else np.empty((0, 5)))

            new_count = count_crossing_objects(tracker_outputs, roi_y1, roi_y2, counted_ids, object_states)
            towel_count += new_count

            for trk in tracker_outputs:
                obj_id = int(trk[4])
                if obj_id in counted_ids:
                    log_writer.writerow([frame_num, obj_id])


            draw_roi_and_boxes(frame, tracker_outputs, roi_y1, roi_y2, counted_ids)
            combined_frame = draw_dashboard_panel(frame, towel_count, frame_num, logo_img)
            out.write(combined_frame)
            cv2.imshow("Fabric Counter", combined_frame)

            if cv2.waitKey(1) == 27:
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
