from mlx_database.dbdata import DBData


class TagData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_all(self):
        sql = "select * from proba_tags where delete_time is null"
        return self.client.query(sql)

    def get_by_app_id(self, app_id):
        sql = "select tag from app_tag_relations where app_id = '{}'".format(app_id)
        tags = self.client.query(sql)
        return [t['tag'] for t in tags]

    def save_app_tag_relation(self, app_id, tag):
        sql = "insert into app_tag_relations (app_id, tag) values ('{}', '{}')".format(app_id, tag)
        return self.client.update(sql) == 1
