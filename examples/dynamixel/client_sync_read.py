import time
import tqdm
import numpy as np
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable

client = DynamixelXLSeriesClient()

motor_ids = [1, 2, 3, 4, 5, 6]
mean_elapsed = 0
N = 1000
for i in tqdm.tqdm(range(N)):
    start = time.time()
    data = client.sync_read(
        motor_ids=motor_ids, control_table=ControlTable.PresentPosition
    )
    end = time.time()
    mean_elapsed += (end - start) * 1000.0 / N
    print(np.array(data) * 360.0 / 4096)

print(mean_elapsed)
client.close()
