import hashlib
import logging
import ssl
import time
import json
import requests

ssl._create_default_https_context = ssl._create_unverified_context


class DspUtil:
    def __init__(self, dsp_params):
        self.dsp_ip = dsp_params['host']
        self.dsp_port = dsp_params['port']
        self.dsp_account_name = dsp_params['account_name']
        self.dsp_account_pwd = dsp_params['account_pwd']
        self.dsp_req_pref = "https://{}:{}".format(self.dsp_ip, self.dsp_port)
        self.auth_url = "/dsp/api/auth/login"
        self.overdue = 0
        self.seed = dsp_params['seed']
        self.tokenId = None
        self.accountId = None

    def req_dsp(self, url, param):
        url = self.dsp_req_pref + url
        self.get_token()
        header = self.create_header()
        res = self.post_data(url, param, header)
        logging.info("url={},  res={}".format(url, json.dumps(res)))
        if res and res["code"] == 200:
            return res["data"]
        return None

    def get_token(self):
        client_time = int(time.time())
        if client_time > self.overdue:  # 请求参数body
            auth_param = {"accountName": self.dsp_account_name, "accountPwd": self.dsp_account_pwd}
            auth_req_url = self.dsp_req_pref + self.auth_url
            auth_res = self.post_data(auth_req_url, auth_param)
            # logging.info("r.text:"+str(auth_res.text))
            logging.info("param={}, url={}, res={}".format(auth_param, auth_req_url, str(auth_res)))
            if auth_res['code'] == 200:
                self.overdue = client_time + 290  # 默认5分钟有效
                data_list = auth_res["data"][0]
                self.tokenId = data_list["tokenId"]
                self.accountId = data_list["accountId"]
                # return {"tokenId": self.tokenId, "accountId": self.accountId}

    def create_header(self):
        client_time = int(time.time())
        sign_str = self.accountId + str(client_time) + self.seed
        sign = self.md5_encode(sign_str)
        head = {"timeStamp": str(int(time.time())), "sign": sign}
        header = dict({"tokenId": self.tokenId, "accountId": self.accountId}, **head)
        return header

    @staticmethod
    def md5_encode(data):
        m = hashlib.md5()
        m.update(data.encode(encoding='gb2312'))
        return m.hexdigest()

    def post_data(self, uri, data, header_param={}, **kwargs):
        headers = {'Content-Type': 'application/json; charset=UTF-8'}
        if header_param:
            headers = dict(headers, **header_param)
        logging.info("send http request. method: POST, uri: {}, kwargs: {}".format(uri, kwargs))
        return requests.post(uri, data=json.dumps(data), verify=False, headers=headers, **kwargs).json()
