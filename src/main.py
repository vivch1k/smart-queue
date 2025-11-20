import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np

from utils.timer import FPSBasedTimer
from utils.draw import draw_cash_info, draw_global_panel, CurvesPlotUpdate
from utils.fuzzy import CashWorkload
from utils.logs import save_logs


ZONE_POLYGON = [
    np.array([[19, 356], [108, 311], [418, 454], [335, 537]]),
    np.array([[228, 251], [279, 265], [459, 331], [400, 393], [197, 313], [186, 269]]),
    np.array([[343, 194], [382, 210], [535, 265], [492, 308], [316, 247], [305, 212]]),
    np.array([[420, 156], [452, 168], [569, 195], [572, 236], [558, 251], [401, 195], [395, 168]])
]

CASH_POINTS = [
    np.array([58, 310]),
    np.array([191, 245]),
    np.array([307, 193]),
    np.array([403, 154])
]


COLOR_ZONE = sv.Color.from_hex("#FF0000")
COLOR_SERVICE = sv.Color.from_hex("#028C00")
COLOR_WAIT = sv.Color.from_hex("#FFB800")

A = np.arange(0, 9) # кол-во людей
a_term_l = (0, 0, 3)
a_term_m = (0, 3, 8)
a_term_h = (3, 8, 8)

B = np.arange(0, 121, 1) # время обслуживания
b_term_l = (0, 0, 60)
b_term_m = (0, 60, 120)
b_term_h = (60, 120, 120)

rules = np.array([
    [0.05, 0.15, 0.35],
    [0.4, 0.7, 0.9],
    [0.8, 1.2, 1.4]
])

video_path = "data/videos/video5.2.mp4"
output_path = "predict/video5.2.mp4"
model_path = "models/yolo11l_fn50img.pt"
logs_path = "data/artifacts/day.csv"


PREDICT = False

if __name__ == "__main__":

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    graph_w = 250 * 3
    final_size = (w + graph_w, h)

    if PREDICT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps=fps, frameSize=final_size)

    model = YOLO(model_path).to("cuda")
    tracker = sv.ByteTrack(lost_track_buffer=5*fps, minimum_matching_threshold=0.85, frame_rate=fps)
    cash_workload = CashWorkload(A, B, a_term_l, a_term_m, a_term_h, b_term_l, b_term_m, b_term_h, rules)

    person_state = {} # {track_id: state}

    zones = [sv.PolygonZone(polygon=zone, triggering_anchors=(sv.Position.CENTER,)) for zone in ZONE_POLYGON]

    service_color_annotator = sv.ColorAnnotator(color=COLOR_SERVICE)
    wait_color_annotator = sv.ColorAnnotator(color=COLOR_WAIT)

    service_label_annotator = sv.LabelAnnotator(color=COLOR_SERVICE, text_scale=0.25, text_padding=3, smart_position=True, text_position=sv.Position.TOP_CENTER)
    wait_label_annotator = sv.LabelAnnotator(color=COLOR_WAIT, text_scale=0.25, text_padding=3, smart_position=True, text_position=sv.Position.TOP_CENTER)

    curves_plot_update = CurvesPlotUpdate(fps)

    print(f"fps = {fps}")
    timers = [FPSBasedTimer(fps=fps) for _ in zones]
    frames = []
    curr_frame = 0
    
    load_zones = [[] for _ in zones]
    service_zones = [[] for _ in zones]
    people_zones = [[] for _ in zones]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break            

        results = model.predict(
            source=frame,
            conf=0.50,
            iou=0.4,
            save=False,
        )[0]

        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        annotated = frame.copy()


        service_zone_time = [0] * len(zones)
        count_people_zone = [0] * len(zones)

        for zone_idx, zone in enumerate(zones):
            annotated = sv.draw_polygon(scene=annotated,
                                        polygon=zone.polygon, 
                                        color=COLOR_ZONE, 
                                        thickness=1)

            detections_in_zone = detections[zone.trigger(detections)]
            n = len(detections_in_zone)
            count_in_zone = zone.current_count
            service_time = 0

            if n > 0:
                # расстояние до кассы и индекс ближайшего
                centers = detections_in_zone.get_anchors_coordinates(anchor=sv.Position.CENTER)
                distances = np.linalg.norm(centers - CASH_POINTS[zone_idx], axis=1)
                nearest_idx = np.argmin(distances)
                # разделения на "в очереди" и "на кассе"
                mask_wait = np.ones(n, dtype=bool)
                mask_wait[nearest_idx] = False

                service_detections  = detections_in_zone[[nearest_idx]]
                wait_detections = detections_in_zone[mask_wait]


                for det, curr_state in zip((service_detections, wait_detections), ["service", "wait"]):
                    for idx in det.tracker_id:
                        state = person_state.get(idx)
                        if state != curr_state:
                            timers[zone_idx].reset(idx)
                        person_state[idx] = curr_state

                # время обслуживания
                service_time = np.max(timers[zone_idx].tick(service_detections))
                service_zone_time[zone_idx] = service_time
                

                # кол-во
                count_people_zone[zone_idx] = count_in_zone
 
                annotated = service_color_annotator.annotate(scene=annotated,
                                                             detections=service_detections)
                
                annotated = wait_color_annotator.annotate(scene=annotated,
                                                          detections=wait_detections)
            
            cash_point_x, cash_point_y = CASH_POINTS[zone_idx]
            indicator_x, indicator_y = cash_point_x-35, cash_point_y-70
            load = cash_workload.sugeno(count_in_zone, service_time)
            annotated = draw_cash_info(annotated, zone_idx, indicator_x, indicator_y,
                           load, count_in_zone, service_time)
        
        
        load_by_zone = np.array([cash_workload.sugeno(c, s) for c, s in zip(count_people_zone, service_zone_time)])     
        if curr_frame % fps == 0:
            frames.append(curr_frame // fps)
            for i in range(len(load_by_zone)):
                load_zones[i].append(load_by_zone[i].round(2))
                people_zones[i].append(count_people_zone[i])
                service_zones[i].append(service_zone_time[i])
            
        h, w, _ = frame.shape
        annotated = draw_global_panel(annotated, np.max(load_by_zone), np.argmax(count_people_zone))

        curves_load, curves_people, curves_service = curves_plot_update.update(curr_frame, frames, load_zones,
                                                                               people_zones, service_zones)

        annotated = cv2.hconcat([
                    annotated,
                    cv2.resize(curves_load, (250, h)),
                    cv2.resize(curves_people, (250, h)),
                    cv2.resize(curves_service, (250, h)),
                ])

        curr_frame += 1

        if PREDICT:
            out.write(annotated)
        cv2.imshow("smart queue", annotated)
        if cv2.waitKey(delay) == 27:
            break
    
    save_logs(path=logs_path,
              num_zones=len(zones),
              time=frames,
              load=load_zones,
              service=service_zones,
              people=people_zones)

    cap.release()
    cv2.destroyAllWindows()
