import os
import sys
import schedule
import time
import logging
from datetime import datetime
from handle_exception.dingding_exception import DingdingExceptionHandler
from server.psi_distribute import LtvPsiMonitor, AlphaPsiMonitor, JdxPsiMonitor
from utillity.daemon import Daemon
from utillity import config_manager as cm


class PsiMonitor(Daemon):
    def __init__(self):
        super(PsiMonitor, self).__init__()

    def run(self):
        sys.path.insert(0, os.path.abspath("../mlDataMonitor"))
        ltvPsiMonitor = LtvPsiMonitor()
        alphaPsiMonitor = AlphaPsiMonitor()
        jdxPsiMonitor = JdxPsiMonitor()

        # # alpha
        schedule.every().day.at("08:31").do(alphaPsiMonitor.job1)
        schedule.every().day.at("16:01").do(alphaPsiMonitor.job2)
        schedule.every().day.at("18:00").do(alphaPsiMonitor.job3)
        # # ltv
        schedule.every().day.at("08:29").do(ltvPsiMonitor.job1)
        schedule.every().day.at("16:02").do(ltvPsiMonitor.job2)
        schedule.every().day.at("18:02").do(ltvPsiMonitor.job3)
        # # jdx
        schedule.every().day.at("08:32").do(jdxPsiMonitor.job1)
        schedule.every().day.at("16:03").do(jdxPsiMonitor.job2)
        schedule.every().day.at("18:03").do(jdxPsiMonitor.job3)





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
                DingdingExceptionHandler(cm.config['robots_psi']).handle(e)


if __name__ == '__main__':
    PsiMonitor().main()


