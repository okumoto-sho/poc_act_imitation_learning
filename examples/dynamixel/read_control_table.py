import time
import tqdm
from koch11.dynamixel.dynamixel_client import DynamixelXLSeriesClient, ControlTable

client = DynamixelXLSeriesClient()

motor_id = 1
mean_elapsed = {str(elem) : 0.0 for elem in ControlTable}
N = 100

for i in tqdm.tqdm(range(N)):
    for elem in ControlTable:   
        start = time.time()
        data = client.read(motor_id=motor_id, control_table=elem)
        end = time.time()
        mean_elapsed[str(elem)] +=  (end - start) * 1000.0 / N

for key in mean_elapsed:
    print(mean_elapsed[key])        

client.close()