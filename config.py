# import os
# class Config(object):
#     DEBUG = True
#     TESTING = False

# class DevelopmentConfig(Config):
#     SECRET_KEY = "12345"



# config = {
#     'development': DevelopmentConfig,
#     'testing': DevelopmentConfig,
#     'production': DevelopmentConfig
# }

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_KEY_SECRET = os.getenv('API_KEY_SECRET')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')