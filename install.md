1. Create a new Conda environment:
```bash
conda create --name fastapi_env python=3.10.4
```

2. Activate the new environment:
```bash
conda activate fastapi_env
```

3. Install FastAPI and its dependencies:
```bash
pip install fastapi routes pydantic pymongo torch
```

4. Create a requirements.txt file:
```bash
pip freeze > requirements.txt
```

```txt
fastapi==0.68.0
uvicorn==0.15.0
```

```bash
pip install -r requirements.txt
```

```bash
uvicorn main:app --reload

/opt/anaconda3/envs/fastapi_env/bin/python -m uvicorn main:app --reload
```