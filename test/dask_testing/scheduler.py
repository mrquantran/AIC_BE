import asyncio
from dask.distributed import Scheduler

async def run_scheduler():
    scheduler = Scheduler()
    await scheduler.start()
    
    print(f"Scheduler is running at: {scheduler.address}")
    
    try:
        await scheduler.finished()
    finally:
        await scheduler.close()

if __name__ == "__main__":
    asyncio.run(run_scheduler())
