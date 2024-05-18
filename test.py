import csv

import torch
print(torch.cuda.is_available())  # Должно вывести: True
print(torch.cuda.device_count())
