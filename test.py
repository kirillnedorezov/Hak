import requests

token = 'y0_AgAAAABFo8-cAATuwQAAAAEFBEN6AAB3jTw9wsFBlLaU_aN01Wz3pH4rkw'
cloudId = 'b1gp7bo4c9nnr8kga380'
folderId = 'b1g24ict85d7hf6cubru'
computeDefaultZone = 'ru-central1-a'

url = f'https://compute.api.cloud.yandex.net/compute/v1/instances?folderId={folder_id}&zoneId={compute_default_zone}'
headers = {
    'Authorization': f'Bearer {token}',
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f'Ошибка при выполнении запроса: {response.status_code}')