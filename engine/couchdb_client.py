import base64
import time
import aiohttp

username = 'admin'
password = 'password'
ip_address = '54.252.244.93'
port_number = '5984'
db_name = 'number_plate'
url = f'http://{username}:{password}@{ip_address}:{port_number}/{db_name}/_bulk_docs'
headers = {'Content-Type': 'application/json'}


def collect_data(frame, labels, present_number_plates, total_data):
    encoded_frame = base64.b64encode(frame)
    encoded_frame_string = encoded_frame.decode('utf-8')

    for label in labels:
        if label not in present_number_plates and label:
            # data to send to db
            data = {'numberplate': label,
                    'datetime': time.time(),
                    'screenshot': encoded_frame_string}

            total_data.append(data)
        present_number_plates.add(label)

    return present_number_plates, total_data


async def send_to_db(total_data):
    async with aiohttp.ClientSession() as session:
        data = {'docs': total_data}
        async with session.post(url, json=data, headers=headers) as resp:
            if resp.status == 201:
                print('Data successfully sent to couchdb')
            else:
                print(f'Error: {resp.status} - {resp.text}')
