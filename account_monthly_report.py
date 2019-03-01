import findspark

import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
import pyspark.sql.functions as sf

import pandas as pd

class account_monthly_report:
    def __init__(self, tx_file_name):
        findspark.init()
        
        self.sc = pyspark.SparkContext(appName='Acme Accounting Monthly Report Pipeline')
        self.sql = SQLContext(self.sc)

        self.data_path = 'data/'
        self.tx_file_name = tx_file_name
        self.mcc_file_name = 'mcc_data.csv'

        self.df_acme_tx = self.sql.read.csv(self.data_path + self.tx_file_name, inferSchema = True, header = True)
        self.df_mcc = self.sql.read.csv(self.data_path + self.mcc_file_name, inferSchema = True, header = True)

        self.df_tx_type_lookup, self.service_system_type_mapping, self.merchant_business_type_mapping = self.lookup_data_preparation()

    def lookup_data_preparation(self):
        # create transaction type lookup
        list_card_type = ['type01', 'type01', 'type01', 'type01', 'type01', 'type01', 'type01', 'type01', 
                        'type02', 'type02', 'type02', 'type02', 'type02', 'type02', 'type02', 'type02']

        list_country = ['Local', 'Local', 'Local', 'Local', 'Inter', 'Inter', 'Inter', 'Inter', 
                        'Local', 'Local', 'Local', 'Local', 'Inter', 'Inter', 'Inter', 'Inter']

        list_card_band = ['brand01', 'brand02', 'brand03', 'other', 'brand01', 'brand02', 'brand03', 'other', 
                        'brand01', 'brand02', 'brand03', 'other', 'brand01', 'brand02', 'brand03', 'other']

        list_transaction_type_code = ['010001', '010002', '010003', '010004', '010011', '010012', '010013', '010014', 
                        '010021', '010022', '010023', '010024', '010031', '010032', '010033', '010034']

        tx_type_lookup_field = [StructField('card_type_lkp', StringType(), False),
                        StructField('country_lkp', StringType(), False),
                        StructField('card_brand_lkp', StringType(), False),
                        StructField('transaction_type_code_lkp', StringType(), False)]

        tx_type_lookup_schema = StructType(tx_type_lookup_field)

        df_tx_type_lookup = self.sql.createDataFrame(zip(list_card_type, list_country, list_card_band, list_transaction_type_code),
                                schema=tx_type_lookup_schema)

        # create service system type lookup
        service_system_type_mapping = {'payment01':'CPF',
                                    'payment02':'OTH',
                                    'payment03':'OTH',
                                    'payment04':'OTH'}

        # create merchant business type lookup
        merchant_business_type_mapping = {'brand01':'30101',
                                    'brand02':'30102',
                                    'brand03':'30103',
                                    'brand04':'30104'}
        
        return df_tx_type_lookup, service_system_type_mapping, merchant_business_type_mapping

    # get fi_code column
    def with_fi_code(self, df):
        return df.withColumn('fi_code', sf.lit('42'))

    # get date column
    def with_date(self, df):
        return df.withColumn('date', sf.from_unixtime(sf.unix_timestamp(sf.last_day(df.date)), 'yyyy-MM-dd'))

    # get service system type column
    def with_service_system_type(self, df):   
        udf_get_service_system_type = sf.udf(lambda payment_method: self.service_system_type_mapping.get(payment_method, ''),
                                            StringType())
        
        return df.withColumn('service_system_type', udf_get_service_system_type(df['payment_method']))

    # get merchant business type column
    def with_merchant_business_type(self, df):
        udf_get_merchant_business_type = sf.udf(lambda service_system_type, card_brand: 
                                                self.merchant_business_type_mapping.get(card_brand, '') 
                                                if service_system_type == 'CPF'
                                                else '',
                                                StringType())
        
        return df.withColumn('merchant_business_type', udf_get_merchant_business_type(df['service_system_type'],
                                                                                    df['card_brand']))

    # get merchant category code column
    def with_merchant_category_code(self, df, df_lookup):
        df = df.alias('df')
        df_lookup = df_lookup.alias('df_lookup')
        df = df.join(df_lookup, df.merchant_category_id == df_lookup.id, how='left')
        
        udf_get_merchant_category_code_for_cpf = sf.udf(lambda service_system_type, code: '9999'
                                                        if service_system_type == 'CPF' and code == None
                                                        else code,
                                                        StringType())
        
        df = df.withColumn('merchant_category_code', udf_get_merchant_category_code_for_cpf(df['service_system_type'],
                                                                                        df['code']))
        return df.drop('code').drop('id')

    # get transaction type column
    def with_transaction_type(self, df, df_lookup):
        
        udf_get_country_code = sf.udf(lambda card_country_issuer_code: 'Local' 
                                    if card_country_issuer_code == 'A029'
                                    else 'Inter',
                                    StringType())
        
        df = df.withColumn('card_country', udf_get_country_code(df['card_country_issuer_code']))
        df = df.join(df_lookup, (df.card_type == df_lookup.card_type_lkp)
                                &(df.card_brand == df_lookup.card_brand_lkp)
                                &(df.card_country == df_lookup.country_lkp),
                    how='left')
        
        udf_replacing_null = sf.udf(lambda transaction_type_code: '099999' 
                                    if transaction_type_code == None 
                                    else transaction_type_code,
                                    StringType())
        
        df = df.withColumn('transaction_type', udf_replacing_null(df['transaction_type_code_lkp']))
        
        return df.drop('card_type_lkp').drop('card_brand_lkp').drop('country_lkp').drop('transaction_type_code_lkp')

    # get sum amount and transaction count columns
    def with_sum_amount_n_transaction_count(self, df):
        groupby_list = ['date', 'service_system_type', 'transaction_type',
                    'merchant_business_type', 'merchant_category_code']
        
        return df.groupBy(groupby_list).agg(sf.round(sf.sum('amount'), 2).alias('amount'),
                                                    sf.count('amount').alias('number'),
                                                    sf.round(sf.avg('amount'), 2).alias('average_amount'))

    # get terminal average amount range column
    def get_terminal_average_amount_range(self, average_amount):
        if average_amount <= 500:
            return '94560000001'
        elif average_amount <= 1000:
            return '94560000002'
        elif average_amount <= 2000:
            return '94560000003'
        elif average_amount <= 5000:
            return '94560000004'
        elif average_amount <= 10000:
            return '94560000005'
        elif average_amount <= 30000:
            return '94560000006'
        else:
            return '94560000007'
        return 

    def with_terminal_average_amount_range(self, df):
        udf_get_terminal_average_amount_range = sf.udf(self.get_terminal_average_amount_range, StringType())

        return df.withColumn('terminal_average_amount_range', 
                            udf_get_terminal_average_amount_range(df['average_amount']))

if __name__ == '__main__':
    None