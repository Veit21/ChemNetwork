# TODO

### What to do next?

- [x] Implement **MongoDB** with **PyMogno** 
    - [x] Configure MongoDB with **Docker**
    - [x] Implement for API calls --> add datetime as information!
    - [x] Connect to DB @ startup --> make sure that connection works, otherwise raise an error and do not connect. Program should not shut down!
    - [x] Save into boolean 'is_connected_to_db' whether connection worked or not, pass it to the endpoint /generate.
    - [x] For the DB read_all endpoint, fix following error: "pydantic_core._pydantic_core.PydanticSerializationError: Unable to serialize unknown type: <class 'bson.objectid.ObjectId'>"
    - [x] Make sure that endpoints are secure and nothing breaks with API calls even when database is not connected!

- [ ] If the client does not connect, this happens 'silently'! Only visible in the docker console.
- [ ] First, mount local database via docker volume for data to persist --> **NOTE**: So far ends with a server connection timeout upon restart. Fix first!
- [ ] ...then, host MongoDB on some server, e.g. via Atlas (and set environment variables)

- [ ] Enable CUDA support in docker
    - [ ] Fix problem when device does not have NVIDIA card/CUDA support (..couldn't test on a CUDA machine yet.)
- [ ] Metal (Apple) support in docker?

- [ ] Find a nice dataset to play around with