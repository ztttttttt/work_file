import os
import sys
import schedule
import time
import logging
from datetime import datetime
from handle_exception.dingding_exception import DingdingExceptionHandler
from server.feature_monitor import LtvFeatureMonitor, OmegaFeatureMonitor, JdxFeatureMonitor, AlphaFeatureMonitor
from server.pass_monitor import LtvPassMonitor, OmegaPassMonitor, JdxPassMonitor
from utillity.daemon import Daemon
from utillity import config_manager as cm


class DataMonitor(Daemon):
    def __init__(self):
        super(DataMonitor, self).__init__()

    def run(self):
        sys.path.insert(0, os.path.abspath("../mlDataMonitor"))
        ltvFeatureMonitor = LtvFeatureMonitor()
        omegaFeatureMonitor = OmegaFeatureMonitor()
        jdxFeatureMonitor = JdxFeatureMonitor()
        alphaFeatureMonitor = AlphaFeatureMonitor()
        # ltv
        schedule.every().day.at("08:30").do(ltvFeatureMonitor.job1)
        schedule.every().day.at("16:00").do(ltvFeatureMonitor.job2)
        # omega
        schedule.every().day.at("08:40").do(omegaFeatureMonitor.job1)
        schedule.every().day.at("16:10").do(omegaFeatureMonitor.job2)
        # jdx
        schedule.every().day.at("08:50").do(jdxFeatureMonitor.job1)
        schedule.every().day.at("16:20").do(jdxFeatureMonitor.job2)
        # alpha
        schedule.every().day.at("09:00").do(alphaFeatureMonitor.job1)
        schedule.every().day.at("16:30").do(alphaFeatureMonitor.job2)
        #
        # ltvPassMonitor = LtvPassMonitor()
        # omegaPassMonitor = OmegaPassMonitor()
        # jdxPassMonitor = JdxPassMonitor()
        # # ctlPassMonitor = CtlPassMonitor()
        # # ltv
        # schedule.every().day.at("09:10").do(ltvPassMonitor.job1)
        # schedule.every().day.at("16:30").do(ltvPassMonitor.job2)
        # # omega
        # schedule.every().day.at("09:15").do(omegaPassMonitor.job1)
        # schedule.every().day.at("16:35").do(omegaPassMonitor.job2)
        # # jdx
        # schedule.every().day.at("09:20").do(jdxPassMonitor.job1)
        # schedule.every().day.at("16:40").do(jdxPassMonitor.job2)
        # # ctl
        # schedule.every().day.at("09:25").do(ctlPassMonitor.job1)
        # schedule.every().day.at("18:15").do(ctlPassMonitor.job2)
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                DingdingExceptionHandler(cm.config['robots']).handle(e)


if __name__ == '__main__':
    DataMonitor().main()


