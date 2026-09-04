# TODO

A list of my TODOs.

### What to do next?

- [ ] Implement **MongoDB** with **PyMogno** 
    - [x] Configure MongoDB with **Docker**
    - [x] Implement for API calls $`\rarr`$ add datetime as information!
    - [x] Connect to DB @ startup $`\rarr`$ make sure that connection works, otherwise raise an error and do not connect. Program should not shut down!
    - [ ] Save into boolean 'is_connected_to_db' whether connection worked or not, pass it to the endpoint /generate.
    - [ ] If the client does not connect, this happens 'silently'! Only visible in the docker console.
    - [x] For the DB read_all endpoint, fix following error: "pydantic_core._pydantic_core.PydanticSerializationError: Unable to serialize unknown type: <class 'bson.objectid.ObjectId'>"
    - [ ] Make sure that endpoints are secure and nothing breaks with API calls even when database is not connected!
    - [ ] Find out where the database within **Docker** is located.