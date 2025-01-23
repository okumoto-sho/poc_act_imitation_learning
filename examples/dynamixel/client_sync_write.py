import time
import tqdm
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable

client = DynamixelXLSeriesClient()

motor_ids = [1, 2, 3, 4, 5, 6]
N = 1000
mean_elapsed = 0

cur = 0
for i in tqdm.tqdm(range(N)):
    start = time.time()
    data = client.sync_write(
        motor_ids, ControlTable.Led, [cur, cur, cur, cur, cur, cur]
    )
    end = time.time()
    mean_elapsed += (end - start) * 1000.0 / N

    if i % 10 == 0:
        cur += 1
        cur %= 2

    time.sleep(0.01)

print(mean_elapsed)
client.close()
