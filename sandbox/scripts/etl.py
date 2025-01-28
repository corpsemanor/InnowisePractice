import pandas as pd
from rapidfuzz import process, fuzz
from scripts.scripts import find_potential_duplicates, get_shop_working_days_sorted


def filter_outliers(df, column, lower_percentile, upper_percentile):
    """
    Filter outliers in a DataFrame column based on percentile thresholds.

    :param df: DataFrame to filter
    :param column: Column name to filter
    :param lower_percentile: Lower percentile threshold (0-100)
    :param upper_percentile: Upper percentile threshold (0-100)
    :return: Filtered DataFrame
    """
    lower_bound = df[column].quantile(lower_percentile / 100)
    upper_bound = df[column].quantile(upper_percentile / 100)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


def clean_sales_data(sales_df, lower_percentile=1, upper_percentile=99):
    """
    Clean the sales_train dataset by removing invalid data and filtering outliers.

    :param sales_df: sales_train DataFrame
    :param lower_percentile: Lower percentile threshold for outliers
    :param upper_percentile: Upper percentile threshold for outliers
    :return: Cleaned sales_train DataFrame
    """
    sales_df = sales_df[~(sales_df['item_price'] < 0)]

    negative_item_cnt = sales_df[sales_df['item_cnt_day'] < 0]
    negative_item_price = sales_df[sales_df['item_price'] < 0]

    print(f"Negative values in item_cnt_day: {negative_item_cnt.shape[0]}")
    print(f"Negative values in item_price: {negative_item_price.shape[0]}")

    sales_df = filter_outliers(sales_df, 'item_cnt_day', lower_percentile, upper_percentile)
    sales_df = filter_outliers(sales_df, 'item_price', lower_percentile, upper_percentile)

    print(f"sales_train shape after filtering outliers: {sales_df.shape}")
    return sales_df


def merge_sales_data(sales_df, items_df, categories_df, shops_df):
    """
    Merge additional information into the sales_train DataFrame.

    :param sales_df: Cleaned sales_train DataFrame
    :param items_df: Items DataFrame
    :param categories_df: Item categories DataFrame
    :param shops_df: Shops DataFrame
    :return: Enriched sales_train DataFrame
    """
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
    return sales_train_complete


def clean_shops_data(shops_df, sales_df):
    """
    Clean the shops DataFrame by handling potential duplicates.

    :param shops_df: Shops DataFrame
    :param sales_df: Sales DataFrame (to merge shop duplicates)
    :return: Cleaned shops DataFrame and updated sales DataFrame
    """
    potential_duplicates = find_potential_duplicates(shops_df, 'shop_name')

    for pair in potential_duplicates:
        dup1, dup2 = pair[:2]
        shop1 = shops_df[shops_df['shop_name'] == dup1]
        shop2 = shops_df[shops_df['shop_name'] == dup2]

        if not shop1.empty and not shop2.empty:
            main_shop_id = min(shop1['shop_id'].iloc[0], shop2['shop_id'].iloc[0])
            duplicate_shop_id = max(shop1['shop_id'].iloc[0], shop2['shop_id'].iloc[0])

            sales_df.loc[sales_df['shop_id'] == duplicate_shop_id, 'shop_id'] = main_shop_id
            shops_df = shops_df[shops_df['shop_id'] != duplicate_shop_id]

    return shops_df, sales_df


def etl(data, lower_percentile=1, upper_percentile=99):
    """
    Perform dataset cleaning based on specified DQC conclusions for each table in data.

    :param data: A dictionary of DataFrames
    :param lower_percentile: Lower percentile threshold for outliers
    :param upper_percentile: Upper percentile threshold for outliers
    :return: Cleaned data with all DataFrames processed
    """
    cleaned_data = {}

    sales_df = clean_sales_data(data['sales_train'].copy(), lower_percentile, upper_percentile)

    shops_df, sales_df = clean_shops_data(data['shops'].copy(), sales_df)

    items_df = data['items']
    categories_df = data['item_categories']
    sales_train_complete = merge_sales_data(sales_df, items_df, categories_df, shops_df)

    test_df = data['test']
    valid_shops = test_df['shop_id'].unique()
    valid_items = test_df['item_id'].unique()

    sales_train_complete = sales_train_complete[sales_train_complete['shop_id'].isin(valid_shops)]
    sales_train_complete = sales_train_complete[sales_train_complete['item_id'].isin(valid_items)]

    sales_train_complete.to_csv('sales_train_complete.csv', index=False)

    cleaned_data['sales_train'] = sales_train_complete
    cleaned_data['shops'] = shops_df

    return cleaned_data