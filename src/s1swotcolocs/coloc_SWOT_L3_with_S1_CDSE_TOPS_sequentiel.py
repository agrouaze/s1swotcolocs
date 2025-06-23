"""
this script replace the prun (temporarily)
May 2025
A Grouazel
"""

import logging
import os
import argparse
from dateutil import rrule
from s1swotcolocs.coloc_SWOT_L3_with_S1_CDSE_TOPS import treat_one_day_wrapper
import sys
import datetime


def setup_logging(log_level=logging.INFO):
    """
    Sets up a standardized logger.

    Args:
        log_level (int): The minimum level of messages to log.
                         Example: logging.DEBUG, logging.INFO, logging.WARNING.
    """
    # Get a logger for this specific module (__name__)
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # If handlers are already configured, don't add them again
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a handler to write logs to the console (standard error)
    handler = logging.StreamHandler(sys.stdout)  # or sys.stderr
    handler.setLevel(log_level)

    # Create a formatter to define the log message format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add the formatter to the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def parse_yyyymmdd(s):
    # logging.debug('s = %s',s)
    try:
        return datetime.datetime.strptime(s, "%Y%m%d")
    except ValueError:
        # raise argparse.ArgumentTypeError(f"Invalid date format: '{s}'. Expected format is YYYYMM.")
        raise argparse.ArgumentTypeError(
            "Invalid date format: '{}'. Expected format is YYYYMMDD.".format(s)
        )


def main():
    # root = logging.getLogger()
    # if root.handlers:
    #     for handler in root.handlers:
    #         root.removeHandler(handler)
    # import argparse
    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--startdate",
        help="YYYYMMDD start SWOT L3 Ifremer collection starts 20230328 ",
        required=True,
        type=parse_yyyymmdd,
    )
    parser.add_argument(
        "--stopdate", help="YYYYMMDD stop", required=True, type=parse_yyyymmdd
    )
    parser.add_argument(
        "--outputdir",
        help="path where the metadata coloc files (.nc) will be saved. [default=computed on the fly from input arguments]",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--confpath", help="path of the config.yml you want to use", required=True
    )
    args = parser.parse_args()

    if args.verbose:
        # logging.basicConfig(
        #     level=logging.DEBUG,
        #     format="%(asctime)s %(levelname)-5s %(message)s",
        #     datefmt="%d/%m/%Y %H:%M:%S",
        # )
        logger = setup_logging(logging.DEBUG)
    else:
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s %(levelname)-5s %(message)s",
        #     datefmt="%d/%m/%Y %H:%M:%S",
        # )
        logger = setup_logging(logging.INFO)

    logger.info("start loops")
    for mode in ["IW", "EW"]:
        logger.info("treat %s", mode)
        for dd in rrule.rrule(rrule.DAILY, dtstart=args.startdate, until=args.stopdate):

            # # example of input line: 20250201 IW /tmp/
            outd = os.path.join(args.outputdir, mode)
            # uu2 = '%s %s %s \n'%(dd.strftime('%Y%m%d'),mode,outd)
            # fid.write(uu2)

            cpt = treat_one_day_wrapper(
                day2treat=dd.strftime("%Y%m%d"),
                outputdir=outd,
                mode=mode,
                disable_tqdm=True,
                confpath=args.confpath,
            )
            logger.info("cpt: %s %s", type(cpt), cpt)
            logger.info("day : %s %s counters:", dd, mode)
            for uu in cpt:
                logger.info("\t %s: %s", uu, cpt[uu])
            # lines.append(uu2)
            # logging.debug(' %s',uu2)
    logger.info("je termine la loop dans %s", os.path.basename(__file__))


if __name__ == "__main__":
    main()
