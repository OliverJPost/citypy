import osmnx

from citypy.logging_setup import logger
from config import cfg


def setup_overpass_api():
    def check_and_set_attr(attr_name, value):
        if not hasattr(osmnx.settings, attr_name):
            raise AttributeError(
                f"The attribute '{attr_name}' is missing in osmnx.settings."
            )
        setattr(osmnx.settings, attr_name, value)

    api_cfg = cfg.overpass_api
    check_and_set_attr("requests_timeout", api_cfg.timeout_seconds)
    try:
        check_and_set_attr("rate_limiting", not cfg.custom_api_instance.enable)
    except AttributeError:
        check_and_set_attr("overpass_rate_limit", not cfg.custom_api_instance.enable)
    if cfg.custom_api_instance.enable:
        logger.info(
            f"Using custom Overpass instance. {cfg.custom_api_instance.endpoint}"
        )
        try:
            check_and_set_attr("overpass_url", cfg.custom_api_instance.endpoint)
        except AttributeError:
            check_and_set_attr("overpass_endpoint", cfg.custom_api_instance.endpoint)
    else:
        logger.info("Using default Overpass instance.")
    check_and_set_attr("use_cache", api_cfg.use_cache)
    if api_cfg.cache_folder:
        check_and_set_attr("cache_folder", api_cfg.cache_folder)
    check_and_set_attr("log_file", api_cfg.output_log)
    check_and_set_attr("log_filename", api_cfg.log_filename)
    try:
        check_and_set_attr("log_folder", api_cfg.log_folder)
    except AttributeError:
        check_and_set_attr("logs_folder", api_cfg.log_folder)
