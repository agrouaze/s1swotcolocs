#!/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python
"""

CE SCRIPT NE SERT A RIEN POUR LE MOEMENT CAR DGX PAS ACCESSIBLE VIA PRUN ET SEULS NEOUDS AVEC ACCES WWWW


"""
import sys
import datetime
print(sys.executable)
import logging
import os
import argparse
import subprocess
import getpass
import calendar
from dateutil import rrule


def end_of_month(yyyymm_str):
    """
    return a date a the very begining of the next month
    """
    # Parse the input string
    year = int(yyyymm_str[:4])
    month = int(yyyymm_str[4:6])

    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]

    # Return datetime object at the end of the day
    # return
    endofmonth = datetime.datetime(year, month, last_day, 23, 59, 59)
    beginingofnextmonth = endofmonth+datetime.timedelta(hours=1)
    return beginingofnextmonth

def parse_yyyymmdd(s):
    # logging.debug('s = %s',s)
    try:
        return datetime.datetime.strptime(s, "%Y%m%d")
    except ValueError:
        # raise argparse.ArgumentTypeError(f"Invalid date format: '{s}'. Expected format is YYYYMM.")
        raise argparse.ArgumentTypeError("Invalid date format: '{}'. Expected format is YYYYMMDD.".format(s))

def main():
    # root = logging.getLogger()
    # if root.handlers:
    #     for handler in root.handlers:
    #         root.removeHandler(handler)
    #import argparse
    parser = argparse.ArgumentParser(description="start prun")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--startmonth",
                        help='YYYYMMDD start SWOT L3 Ifremer collection starts 20230328 ',required=True,type=parse_yyyymmdd,)
    parser.add_argument("--stopmonth",
                        help='YYYYMMDD stop', required=True,type=parse_yyyymmdd)
    parser.add_argument('--outputdir',
                        help='path where the metadata coloc files (.nc) will be saved. [default=computed on the fly from input arguments]',
                        required=True,default=None)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%d/%m/%Y %H:%M:%S",
        )
    prunexe = "/appli/prun/bin/prun"
    username = getpass.getuser()
    logging.info('prunexe : %s',prunexe)
    ontheflymodifiedlisting = os.path.join('/home1/scratch/',username,'tmp_listing_s1_swot_metadata_colo_files_production.txt')
    fid =open(ontheflymodifiedlisting,'w')
    lines = []
    for dd in rrule.rrule(rrule.DAILY,dtstart=args.startmonth,until=args.stopmonth):
        for mode in ['IW','EW']:
            # # example of input line: 20250201 IW /tmp/
            outd = os.path.join(args.outputdir,mode)
            uu2 = '%s %s %s \n'%(dd.strftime('%Y%m%d'),mode,outd)
            fid.write(uu2)
            lines.append(uu2)
            logging.debug(' %s',uu2)

    fid.close()
    logging.info('temporary listing updated : %s',ontheflymodifiedlisting)
    cpt = len(lines)
    logging.info('number of lines in the job array : %i',cpt)

    pbs = os.path.abspath(os.path.join(os.path.dirname(__file__),'coloc_SWOT_L3_with_S1_CDSE_TOPS.pbs'))
    logging.info('pbs : %s',pbs)
    assert os.path.exists(pbs)
    # call prun
    # logging.info('pbs = %s',pbs)
    opts = " --split-max-lines=50 -e "
    py2 = "/home1/datawork/agrouaze/conda_envs2/envs/py2.7_cwave/bin/python "
    cmd = py2 + prunexe + opts + pbs + " " + ontheflymodifiedlisting
    logging.info("cmd to cast = %s", cmd)
    st = subprocess.check_call(cmd, shell=True)
    logging.info("status cmd = %s", st)


if __name__ == "__main__":
    main()

