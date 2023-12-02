import couchdb
import base64
import time

username = 'admin'
password = 'password'
ip_address = '54.252.244.93'
port_number = '5984'
db_name = 'number_plate'
file_type = 'image/jpeg'

couch = couchdb.Server(
    f"http://{username}:{password}@{ip_address}:{port_number}")

if db_name in couch:
    db = couch[db_name]
else:
    db = couch.create(db_name)


def collect_data(frame, labels, present_number_plates, total_data):
    encoded_frame = base64.b64encode(frame)
    encoded_frame_string = '' + str(encoded_frame)

    for label in labels:
        if label not in present_number_plates and not label:
            print(label, present_number_plates)
            # data to send to db
            data = {'numberplate': label,
                    'datetime': time.time(),
                    'screenshot': encoded_frame_string}

            total_data.append(data)
        present_number_plates.add(label)

    return present_number_plates, total_data


def send_to_db(total_data):
    db.update(total_data)
    print('Number plates sent to db')
