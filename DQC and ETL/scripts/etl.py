import pandas as pd
from rapidfuzz import process, fuzz
from scripts.scripts import *
import pandas as pd
from scripts.scripts import find_potential_duplicates, get_shop_working_days_sorted

def etl(data):
    """
    Perform dataset cleaning based on specified DQC conclusions for each table in data.
    Focused on sales_train dataset cleaning.

    :param data: A dictionary of DataFrames
    :return: Cleaned data with all DataFrames processed
    """
    cleaned_data = {}

    sales_df = data['sales_train'].copy()

    sales_df = sales_df[~((sales_df['item_price'] < 0) & (sales_df['item_cnt_day'] >= 0))]

    negative_item_cnt = sales_df[sales_df['item_cnt_day'] < 0]
    negative_item_price = sales_df[sales_df['item_price'] < 0]

    print(f"Negative values in item_cnt_day: {negative_item_cnt.shape[0]}")
    print(f"Negative values in item_price: {negative_item_price.shape[0]}")

    percs = sales_df[['item_cnt_day', 'item_price']].describe(percentiles=[0.01, 0.99])

    lower_bound_item_cnt = percs.loc['1%', 'item_cnt_day']
    upper_bound_item_cnt = percs.loc['99%', 'item_cnt_day']
    lower_bound_item_price = percs.loc['1%', 'item_price']
    upper_bound_item_price = percs.loc['99%', 'item_price']

    sales_df = sales_df[
        (sales_df['item_cnt_day'] >= lower_bound_item_cnt) & 
        (sales_df['item_cnt_day'] <= upper_bound_item_cnt) | 
        (sales_df['item_cnt_day'] < 0) 
    ]
    
    sales_df = sales_df[
        (sales_df['item_price'] >= lower_bound_item_price) & 
        (sales_df['item_price'] <= upper_bound_item_price)
    ]

    print(f"sales_train shape after filtering outliers: {sales_df.shape}")

    shops_df = data['shops'].copy()
    potential_duplicates = find_potential_duplicates(shops_df, 'shop_name')

    for pair in potential_duplicates:
        dup1, dup2 = pair[:2]
        shop1 = shops_df[shops_df['shop_name'] == dup1]
        shop2 = shops_df[shops_df['shop_name'] == dup2]
        
        if not shop1.empty and not shop2.empty:
            shop_id = shop2['shop_id'].iloc[0]
            shop_name = shop2['shop_name'].iloc[0]
            
            shops_df.loc[shops_df['shop_name'] == dup1, 'shop_id'] = shop_id
            shops_df.loc[shops_df['shop_name'] == dup1, 'shop_name'] = shop_name

    cleaned_data['shops'] = shops_df

    items_df = data['items']
    categories_df = data['item_categories']

    sales_train_complete = sales_df.merge(items_df[['item_id', 'item_category_id']], on='item_id', how='left')
    sales_train_complete = sales_train_complete.merge(items_df[['item_id', 'item_name']], on='item_id', how='left')
    sales_train_complete = sales_train_complete.merge(categories_df[['item_category_id', 'item_category_name']], on='item_category_id', how='left')
    sales_train_complete = sales_train_complete.merge(shops_df[['shop_id', 'shop_name']], on='shop_id', how='left')

    sales_train_complete['date'] = pd.to_datetime(sales_train_complete['date'], format='%d.%m.%Y')

    sales_train_complete['date__day'] = (sales_train_complete['date'] - min(sales_train_complete['date'])).dt.days
    sales_train_complete['date__week'] = sales_train_complete['date__day'] // 7
    sales_train_complete['date__day_of_month'] = sales_train_complete['date'].dt.day
    sales_train_complete['date__day_of_week'] = sales_train_complete['date'].dt.dayofweek
    sales_train_complete['date__week_of_year'] = sales_train_complete['date'].dt.isocalendar().week
    sales_train_complete['date__month_of_year'] = sales_train_complete['date'].dt.month
    sales_train_complete['date__year'] = sales_train_complete['date'].dt.year

    sales_train_complete.rename({'date_block_num': 'date__month'}, inplace=True, axis=1)

    test_df = data['test']
    valid_shops = test_df['shop_id'].unique()
    valid_items = test_df['item_id'].unique()

    sales_train_complete = sales_train_complete[sales_train_complete['shop_id'].isin(valid_shops)]
    sales_train_complete = sales_train_complete[sales_train_complete['item_id'].isin(valid_items)]

    sales_train_complete.to_csv('sales_train_complete.csv', index=False)

    cleaned_data['sales_train'] = sales_train_complete

    return cleaned_data
