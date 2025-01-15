import pandas as pd
from rapidfuzz import process, fuzz

cities_list = ['Москве', 'Санкт-Петербурге', 'Казани', 'Новосибирске']


def get_shop_working_days_sorted(sales_data, shops_data):
    """
    Возвращает отсортированный словарь с названиями магазинов и количеством дней, 
    в которые каждый магазин был активен (уникальные даты), отсортированный по возрастанию.
    
    :param sales_data: pandas.DataFrame, данные о продажах (sales_train)
    :param shops_data: pandas.DataFrame, данные о магазинах (shops)
    :return: dict, отсортированный словарь {название магазина: количество уникальных дней}
    """
    sales_data['date'] = pd.to_datetime(sales_data['date'], dayfirst=True)
    grouped = sales_data.groupby('shop_id')['date'].nunique()
    
    shop_names = dict(zip(shops_data['shop_id'], shops_data['shop_name']))
    
    shop_working_days = {shop_names.get(shop_id, "Unknown Shop"): days for shop_id, days in grouped.items()}
    sorted_working_days = dict(sorted(shop_working_days.items(), key=lambda item: item[1]))
    
    return sorted_working_days

def find_potential_duplicates(df, column_name, similarity_threshold=85):
    """
    Finds potential duplicates in specified column in dataframe
    расстояние Левенштейна
    :param df: pandas.DataFrame, input dataframe
    :param column_name: str, column name for dataframe
    :param similarity_threshold: int, similarity treshold
    :return: pairs list of duplicates with similarity score
    """
    unique_values = df[column_name].unique()
    
    duplicates = []
    
    for value in unique_values:
        matches = process.extract(value, unique_values, scorer=fuzz.ratio, limit=None)
        for match in matches:
            other_value, score = match[0], match[1]
            if score >= similarity_threshold and value != other_value:
                duplicates.append((value, other_value, score))
    
    unique_duplicates = set(tuple(sorted(pair[:2])) + (pair[2],) for pair in duplicates)
    
    return list(unique_duplicates)

def unique_nan_total(data):
    """
    Checks dataframe for unique values existence, sum of every value, and Nan values.

    :param data: your Dataframe
    :return: Dataframe report table
    """

    report = []

    for table_name, df in data.items():
        for column in df.columns:
            unique_count = df[column].nunique()
            nan_count = df[column].isna().sum()
            total_count = len(df[column])
            dtype = df[column].dtype
            
            report.append({
                'table_name': table_name,
                'column_name': column,
                'unique_count': unique_count,
                'nan_count': nan_count,
                'total_count': total_count,
                'dtype': dtype
            })
    
    report_df = pd.DataFrame(report)
    return report_df

def timeline_check(data):

    """
    Checks timelines for gaps, finds Start date and end date to ensure timeline consistancy.

    :param data: yor Dataframe 
    """
    dates_formatted = []
    dates = list(data['sales_train']['date'])
    for date in dates:
        date = date.replace('.','-')
        dates_formatted.append(date)
    
    dates_formatted = pd.to_datetime(dates_formatted, dayfirst=True)
    first_date = dates_formatted.min()
    last_date = dates_formatted.max()

    unique_dates = list(set(dates_formatted))
    full_date_range = pd.date_range(start=first_date, end=last_date, freq='D')

    missing_dates = full_date_range.difference(unique_dates)
    
    print(f'First date in timeline: {first_date}, last date in timeline: {last_date}\n')
    print('Date skipped:', list(missing_dates)) if len(list(missing_dates)) != 0 else print('No dates skipped.\n')

def min_max_mean(data):
    """
    Perform DataFrame check for minimum, maximum, and mean value to help detect outliers,
    excluding rows with 'id' in their index.

    :param data: Dictionary of DataFrames.
    :return: Cleaned DataFrame report table.
    """
    import pandas as pd

    all_stats = []

    for table_name, df in data.items():
        df.columns = df.columns.map(str)
        
        numeric_df = df.select_dtypes(include='number')
        stats = numeric_df.agg(['mean', 'max', 'min']).transpose()
        stats['table_name'] = table_name
        all_stats.append(stats)

    combined_stats = pd.concat(all_stats, axis=0).reset_index()

    cleaned_stats = combined_stats[~combined_stats['index'].str.contains('id', case=False)]

    return cleaned_stats

def check_for_skips(_list):
    """
    Checks list for skipped values to help understand data consistancy and detect outliers.

    :param _list: List of values
    :return: List of skipped values
    """
    skips = []
    unique_sorted_list = set(list(sorted(_list)))

    for val in range(max(unique_sorted_list)):
        if val not in unique_sorted_list:
            skips.append(val)
            
    return skips

def get_negative_rows_by_column(df):

    """
    Returns negative rows by column from datafrme, in our data should not be any negative values,
    so it'll help detect outliers.

    :param df: your Dataframe
    :return: Dataframe consist of only negative vals
    """
    negative_rows = df[df < 0]
    return negative_rows


def get_negative_rows(df):
    """
    Returns negative rows by from datafrme, in our data should not be any negative values,
    so it'll help detect outliers.

    :param df: your Dataframe
    :return: Dataframe w/o positive rows
    """
    negative_rows = df[(df.select_dtypes(include='number') < 0).any(axis=1)]
    return negative_rows

def count_elements(lst):
    """
    Counts elements in list, finds amount of every element repeatence, helps to find dependencies in data.

    :param lst: List of values (column form DF) 
    :return: dictionary of value: repeat amount 
    """
    element_count = {}
    for element in lst:
        if element in element_count:
            element_count[element] += 1
        else:
            element_count[element] = 1
    return element_count

def get_item_names(items_df, top_returns_dict):
    """
    Used for convert top reurns dict of itrm_id:amount of returns to item_name:amount of returns

    :param items_df: Dataframe of items and their ids
    :param top_returns_dict: Dictionary of itrm_id:amount 
    """
    filtered_items = items_df[items_df['item_id'].isin(top_returns_dict.keys())]
    return filtered_items['item_name'].tolist()

def date_corrector(data):
    """
    Used for converting dates from df to usable format.

    :param data: your Dataframe
    :return: Datafrme w fixed dates
    """
    dates = data['sales_train']['date']
    dates = pd.to_datetime(list(dates), dayfirst=True)
    
    return dates

def shops_sales(data):
    """
    Creates report to show amount of summary saler for every shop over all time.

    :param data: your Dataframe
    :return: Dictionary of shop_id:amount of sales
    """
    report = {}
    shops = data['sales_train']['shop_id']
    for element in shops:
        if element in report:
            report[element] += 1
        else:
            report[element] = 0
    return report

def sales_by_date(data):
    """
    Shows summary sales for every date in dataset

    :return: Dictionary of date: amount of sales
    """
    
    report = {}
    dates = date_corrector(data)
    for date in dates:
        if date in report:
            report[date] += 1
        else:
            report[date] = 0
    return report


def find_unique_items(df):
    """
    Creates
    """
    unique_shop_ids = df['shop_id'].unique()
    all_item_ids = set(df['item_id'].unique())

    result = {}

    for shop_id in unique_shop_ids:
        items_in_shop = set(df[df['shop_id'] == shop_id]['item_id'].unique())
        never_sold_items = all_item_ids - items_in_shop
        result[shop_id] = never_sold_items

    return result