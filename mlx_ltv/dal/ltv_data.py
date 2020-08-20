from mlx_database.dbdata import DBData


class LtvData(DBData):
    def __init__(self, client, table=None):
        super().__init__(client, table)

    def get_by_app_id(self, app_id):
        app_id = str(app_id).upper()
        sql = """
            select a.Principal+0E0 as principal, a.Repayments as repayments,
                   ua.TotalCredits+0E0 as total_credits, ua.AvailableCredits+0E0 as available_credits,
                   ua.MonthlyPaybackCredits+0E0 as monthly_payback_credits, ua.UserId as user_id,
                   ui.Income+0E0 as income,
                   ur.IdName as name ,ur.IdNumber as id_number, ur.Mobile as mobile
            from clapplicationobjects as a
            join cluseraccountobjects as ua on a.UserId = ua.UserId
            join cluserinfoobjects as ui on a.UserId = ui.UserId
            join cluserredundancydataobjects as ur on a.UserId = ur.UserId
            where a.AppId = '{}'
            order by ui.LastModified desc
        """.format(app_id)
        result = self.client.query(sql)
        return result[0] if result else None

    def get_user_repayment_times(self, user_id):
        user_id = str(user_id).upper()
        sql = """
            select 1 from clapplicationobjects
            where UserId = '{}' and RepaymentStatus in (500, 600)
        """.format(user_id)
        result = self.client.query(sql)
        return len(result)

    def get_opscore_by_app_id(self, app_id):
        app_id = str(app_id).upper()
        sql = """
            select c.score as op_score, c.model as op_model, c.level as op_level
            from clapplicationobjects as a
            join mcl_user_credit as c
            on a.UserId = c.userId
            where a.AppId = '{}'
        """.format(app_id)
        result = self.client.query(sql)
        return result[0] if result else None
