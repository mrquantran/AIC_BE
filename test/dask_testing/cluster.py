import torch
import dask
from dask.distributed import Client, LocalCluster
import torch.nn as nn
import multiprocessing
import numpy as np

class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Expects input of size 1, outputs size 1

    def forward(self, x):
        return self.linear(x)

def run_model(data):
    model = YourModel()
    model.load_state_dict(torch.load('your_model.pth'))
    model.eval()
    with torch.no_grad():
        # Ensure the input is 2D: (batch_size, input_size)
        input_tensor = torch.tensor([[data]], dtype=torch.float32)
        return model(input_tensor).item()  # Return a scalar

def main():
    # Set up Dask cluster
    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)
    print(client.dashboard_link)

    # Generate example data - now just single values
    data = np.random.randn(10000)

    # Distribute computations
    futures = client.map(run_model, data)
    results = client.gather(futures)

    print(results)

    # Close the client when done
    client.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()  # For Windows compatibility
    main()
