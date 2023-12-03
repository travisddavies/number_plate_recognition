import aiohttp

username = 'admin'
password = 'password'
ip_address = '54.252.244.93'
port_number = '5984'
db_name = 'number_plate'
url = f'http://{username}:{password}@{ip_address}:{port_number}/{db_name}/_bulk_docs'
headers = {'Content-Type': 'application/json'}




async def send_to_db(total_data):
    async with aiohttp.ClientSession() as session:
        data = {'docs': total_data}
        async with session.post(url, json=data, headers=headers) as resp:
            if resp.status == 201:
                print('Data successfully sent to couchdb')
            else:
                print(f'Error: {resp.status} - {resp.text}')
