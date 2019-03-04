import findspark

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
import os
import pyspark.sql.functions as sf

import pandas as pd

from account_monthly_report import AccountMonthlyReport

findspark.init()

sc = pyspark.SparkContext(appName='Acme Accounting Monthly Report Pipeline',
                          pyFiles=[os.path.join(os.path.abspath(os.path.dirname(__file__)), 'account_monthly_report.py')])
sql = SQLContext(sc)

amr = AccountMonthlyReport('acme_december_2018.csv', sql)
df = amr.generate_report_dataframe()
# amr.write_csv_report()

sc.stop