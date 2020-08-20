import requests
import traceback


class DingdingExceptionHandler(object):
    def __init__(self, robots):
        self.robots = robots

    def handle(self, ex=None, msg=None):
        msgs = []
        if msg:
            if isinstance(msg,list):
                msgs.extend(msg)
            else:
                msgs.append(msg)
        if ex:
            msgs.append(traceback.format_exc())
        msg_dict = {
            "msgtype": "text",
            "text": {
                "content": '\n'.join(msgs)
            }
        }
        for rob in self.robots:
            requests.post(rob, json=msg_dict)

