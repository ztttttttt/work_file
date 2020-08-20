from mlx_http.http_manager import get_json


class AppData:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def get_all_id_by_req_id(self, req_id):
        """
        :param req_id: same as app_id in ML
        :return:
            {
                "appId": "0",
                "requestId": "111111",
                "userId": "111111",
                "userAppId": ""
            }
        """
        uri = "http://{}:{}/ctl-application/application/{}/all-ids?type=1".format(self.host, self.port, req_id)
        return get_json(uri)

    def get_extend_info(self, req_id):
        """
        :return:
            {
                "longitude": "114.1111",
                "latitude": "45.22222",
                "marsLongitude": "114.1111",
                "marsLatitude":"45.22222",
                "ip":"114.12.12.145",
                "bssid":"1111eddddfff",
                "deviceToken":"maidanxia-sssds44444",
                "deviceTokenType":"H5"
            }
        """
        uri = "http://{}:{}/ctl-application/application/{}/application-extend-info".format(self.host, self.port, req_id)
        return get_json(uri)

    def sync_td_data_to_application(self, data):
        pass

    def get_carrier_cert(self, req_id):
        """
        :return:
            {
                "appId": "41f092d0-87ee-11e7-a55d-00163e0f04ac",
                "token": "85d27d31b27c4a2fbc462ee753c78e90",
                "mobile": "13012343123",
                "name": "jiaoh",
                "idCard": "372929199305160070",
                "taskId": "30b415c0-9373-11e7-807a-00163e0cb266"
            }
        """
        uri = "http://{}:{}/ctl-application/application/{}/report?type=1".format(self.host, self.port, req_id)
        return get_json(uri)
