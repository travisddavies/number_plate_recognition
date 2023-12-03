import base64
import time
from metrics import get_iou


def collect_data(frame, labels, present_number_plates, total_data):
    encoded_frame = base64.b64encode(frame)
    encoded_frame_string = encoded_frame.decode('utf-8')

    for label in labels:
        index = find_overlapping_bbox(label, bbox)
        if index >= 0:
            update_labels(label, total_data, index, times_on_screen)
        if label not in present_number_plates and label:
            # data to send to db
            data = {'numberplate': label,
                    'datetime': time.time(),
                    'screenshot': encoded_frame_string}

            total_data.append(data)
        present_number_plates.add(label)

    return present_number_plates, total_data


def find_overlapping_bbox(label, bbox, total_data):
    largest_iou = 0
    overlap_index = -1
    for i, data in enumerate(total_data):
        bbox_on_screen = total_data['bbox']
        iou = get_iou(bbox, bbox_on_screen)
        if iou > largest_iou and iou > 0.75:
            largest_iou = iou
            overlap_index = i

    return overlap_index


def update_labels(candidate_label, total_data, overlap_index, times_on_screen):
    overlap = total_data[overlap_index]
    overlapping_label = overlap['label']
    candidate_screen_time = get_time_on_screen(
        candidate_label, times_on_screen)
    overlap_screen_time = get_time_on_screen(
        overlapping_label, times_on_screen)

    if candidate_screen_time > overlap_screen_time:
        total_data[overlap_index]['label'] = candidate_label


def get_time_on_screen(label, times_on_screen):
    if label not in times_on_screen:
        start_time = time.time()
        times_on_screen[label]['start_time'] = start_time
        times_on_screen[label]['end_time'] = start_time
        return 0

    start_time = times_on_screen[label]['start_time']
    end_time = times_on_screen[label]['end_time']
    time_on_screen = end_time - start_time
    times_on_screen[label]['end_time'] = time.time()

    return time_on_screen
