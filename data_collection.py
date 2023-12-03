import base64
import time
from metrics import get_iou


def collect_data(frame, labels, bboxes, total_data, times_on_screen):
    encoded_frame = base64.b64encode(frame)
    encoded_frame_string = encoded_frame.decode('utf-8')

    for label, bbox in zip(labels, bboxes):
        index = find_overlapping_bbox(label, bbox)
        if index >= 0:
            update_time_on_screen(label, times_on_screen)
            update_last_seen(total_data, index)
            update_labels(label, total_data, index, times_on_screen)
        else:
            data = {'numberplate': label,
                    'datetime': time.time(),
                    'screenshot': encoded_frame_string,
                    'bbox': bbox}
            total_data.append(data)
            index = len(total_data) - 1

        update_bbox(total_data, index, bbox)
        clean_up_times(times_on_screen)

    return total_data, times_on_screen


def update_last_seen(total_data, index):
    total_data[index]['datetime'] = time.time()


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
    candidate_screen_time = get_time_on_screen(
        candidate_label, times_on_screen)
    overlap_screen_time = get_time_on_screen(
        total_data[overlap_index]['label'], times_on_screen)

    if candidate_screen_time > overlap_screen_time:
        total_data[overlap_index]['label'] = candidate_label


def get_time_on_screen(label, times_on_screen):
    start_time = times_on_screen[label]['start_time']
    end_time = times_on_screen[label]['end_time']
    time_on_screen = end_time - start_time

    return time_on_screen


def update_time_on_screen(label, times_on_screen):
    start_time = time.time()
    if label not in times_on_screen or \
            start_time - times_on_screen[label]['end_time'] > 30:
        times_on_screen[label]['start_time'] = time.time()

    times_on_screen[label]['end_time'] = time.time()


def update_bbox(total_data, index, bbox):
    total_data[index]['bbox'] = bbox


def clean_up_times(times_on_screen):
    for label in times_on_screen:
        if time.time() - times_on_screen[label]['end_time'] > 5 * 60:
            del times_on_screen[label]
