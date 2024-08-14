from dask.distributed import Worker, Client
import torch
import torch.nn as nn
import asyncio

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
        input_tensor = torch.tensor([[data]], dtype=torch.float32)
        return model(input_tensor).item()

async def run_worker():
    # Connect to the scheduler
    scheduler_address = 'tcp://192.168.1.9:56183'  # Replace with actual scheduler IP if needed
    worker = await Worker(scheduler_address)
    
    # Register the run_model function with the worker
    worker.register_task('run_model', run_model)
    
    print(f"Worker is running, connected to: {worker.scheduler.address}")
    
    await worker.finished()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run_worker())
