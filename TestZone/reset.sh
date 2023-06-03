#! /bin/bash

python3 -c 'from database import *; Base.metadata.drop_all(engine);  Base.metadata.create_all(engine)'