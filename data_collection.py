import base64
import time
import aiohttp
import numpy as np


class NumberPlateCollector:
    def __init__(self):
        self.username = 'admin'
        self.password = 'password'
        self.ip_address = '54.252.244.93'
        self.port_number = '5984'
        self.db_name = 'number_plate'
        self.url = f'http://{self.username}:{self.password}@{self.ip_address}:{self.port_number}/{self.db_name}/_bulk_docs'
        self.headers = {'Content-Type': 'application/json'}
        self.times_on_screen = []
        self.total_data = []

    def collect_data(self, frame, labels, bboxes):
        encoded_frame = base64.b64encode(frame)
        encoded_frame_string = encoded_frame.decode('utf-8')

        for label, bbox in zip(labels, bboxes):
            index = self._find_overlapping_bbox(label, bbox)
            if index >= 0:
                self._update_time_on_screen(label, self.times_on_screen)
                self._update_last_seen(self.total_data, index)
                self._update_labels(label, self.total_data,
                                    index, self.times_on_screen)
            else:
                data = {'numberplate': label,
                        'datetime': time.time(),
                        'screenshot': encoded_frame_string,
                        'bbox': bbox}
                self.total_data.append(data)
                index = len(self.total_data) - 1

            self._update_bbox(index, bbox)
            self._clean_up_times()

    async def send_to_db(self):
        async with aiohttp.ClientSession() as session:
            data = {'docs': self.total_data}
            async with session.post(self.url, json=data, headers=self.headers) as resp:
                if resp.status == 201:
                    await print('Data successfully sent to couchdb')
                    self.total_data = []
                else:
                    await print(f'Error: {resp.status} - {resp.text}')

    def __len__(self):
        return len(self.total_data)

    def _update_last_seen(self, index):
        self.total_data[index]['datetime'] = time.time()

    def _get_iou(bbox1, bbox2):
        # Coordinates of the area of intersection.
        ix1 = np.maximum(bbox1[0], bbox2[0])
        iy1 = np.maximum(bbox1[1], bbox2[1])
        ix2 = np.minimum(bbox1[2], bbox2[2])
        iy2 = np.minimum(bbox1[3], bbox2[3])

        # Intersection height and width.
        i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
        i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

        area_of_intersection = i_height * i_width

        # Ground Truth dimensions.
        gt_height = bbox2[3] - bbox2[1] + 1
        gt_width = bbox2[2] - bbox2[0] + 1

        # Prediction dimensions.
        pd_height = bbox1[3] - bbox1[1] + 1
        pd_width = bbox1[2] - bbox1[0] + 1

        area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

        iou = area_of_intersection / area_of_union

        return iou

    def _find_overlapping_bbox(self, label, bbox):
        largest_iou = 0
        overlap_index = -1

        for i, data in enumerate(self.total_data):
            bbox_on_screen = self.total_data['bbox']
            iou = self._get_iou(bbox, bbox_on_screen)
            if iou > largest_iou and iou > 0.75:
                largest_iou = iou
                overlap_index = i

        return overlap_index

    def _update_labels(self, candidate_label, overlap_index):
        candidate_screen_time = self._get_time_on_screen(
            candidate_label, self.times_on_screen)
        overlap_screen_time = self._get_time_on_screen(
            self.total_data[overlap_index]['label'])

        if candidate_screen_time > overlap_screen_time:
            self.total_data[overlap_index]['label'] = candidate_label

    def _get_time_on_screen(self, label):
        start_time = self.times_on_screen[label]['start_time']
        end_time = self.times_on_screen[label]['end_time']
        time_on_screen = end_time - start_time

        return time_on_screen

    def _update_time_on_screen(self, label):
        start_time = time.time()
        if label not in self.times_on_screen or \
                start_time - self.times_on_screen[label]['end_time'] > 30:
            self.times_on_screen[label]['start_time'] = time.time()

        self.times_on_screen[label]['end_time'] = time.time()

    def _update_bbox(self, index, bbox):
        self.total_data[index]['bbox'] = bbox

    def _clean_up_times(self):
        for label in self.times_on_screen:
            if time.time() - self.times_on_screen[label]['end_time'] > 5 * 60:
                del self.times_on_screen[label]
