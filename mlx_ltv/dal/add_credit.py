from mlx_database.dbdata import DBData


class AddCreditData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_pmt_tot_credit(self, user_id):
        sql = '''
        select TotalCredits ,MonthlyPaybackCredits from cluseraccountobjects where UserId = '{}' order by LastModified desc limit 1
        '''.format(user_id)
        res = self.client.query(sql)
        return (float(res[0]['MonthlyPaybackCredits']), float(res[0]['TotalCredits'])) if res else (None, None)
