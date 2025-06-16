import logging

from yaml import CLoader as Loader
from yaml import load


logger = logging.getLogger("s1swotcolocs.get_config_info")
logger.addHandler(logging.NullHandler())


# def get_config_file_path():
#     # The configuration path is determined in the following order:
#     # 1. First, check the XSARSLC_CONFIG_PATH environment variable if it's set.
#     # 2. If not set, fall back to localconfig.yaml.
#     # 3. If neither is found, default to config.yaml.
#
#     default_local_config_path = os.path.join(
#         os.path.dirname(s1swotcolocs.__file__), "localconfig.yml"
#     )
#     default_config_path = os.path.join(os.path.dirname(s1swotcolocs.__file__), "config.yml")
#     potential_local_config_path = os.environ.get(
#         "XSARSLC_CONFIG_PATH", default_local_config_path
#     )
#
#     if os.path.exists(potential_local_config_path):
#         config_path = potential_local_config_path
#     else:
#         if os.path.exists(default_local_config_path):
#             config_path = default_local_config_path
#         else:
#             config_path = default_config_path
#
#     logger.info("Config path: %s", config_path)
#     return config_path


def get_conf_content(conf_path):
    # stream = open(get_config_file_path(), "r")
    stream = open(conf_path, "r")
    conf = load(stream, Loader=Loader)
    return conf
