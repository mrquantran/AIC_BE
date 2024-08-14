cd to project `backend`

Install the virtual environment using venv
`python3 -m venv <virtual-environment-name>`

Run the virtual environment
`.\<virtual-environment-name>\Scripts\activate`

Install the required libraries
`pip install fastapi routes pydantic pymongo torch`

Start mongoDBCompass app on Windows and run the database
![mongo](https://github.com/user-attachments/assets/81add634-df3c-4db8-a04b-cfca197cae27)

Start the server from the command line!
`fastapi dev .\main.py`

The server currently have 3 routes
GET / to check the database
POST /predict to test the model
POST /uploadsomething to insert an image path {"path": str, value: float}
