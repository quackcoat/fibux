import pandas as pd
import re
import os
import re
import csv
import glob
import tkinter as tk
import warnings
import ast
import calendar
from tkinter import filedialog
from datetime import datetime,timedelta
from prettytable import PrettyTable as pt
from collections import defaultdict
import math as m
from art import *
from pynput.keyboard import Key, Listener
import readline
import traceback


def import_settings(_type):
    if _type == 'categories':
        # create an empty dictionary to store DataFrames for each line
        auto_categories = []
        categories = {}
        # open the text file for reading
        category_input = ''
        with open("settings." + _type) as f:
            current_category = None
            current_subcategory = None
            
            for line in f:
                if line[0] == '*':
                    _category_input = line[1:].strip() 
                elif line[0] != '#':
                    if _category_input == 'Categories':
                        _split = line.split(';')
                        if len(_split) == 2:
                            if _split[0].isdigit():
                                current_category = int(_split[0].strip())
                                categories[current_category] = {'name': _split[1].strip()}
                            else:
                                current_subcategory = _split[0].strip()
                                categories[current_category][current_subcategory] = _split[1].strip()
                    elif _category_input == 'Auto category':
                        change_values = {}
                        search_criteria = {}
                        _split = line.split(';')
                        auto_categories.append({'change':{},'search':{}})
                        for _spi in _split:
                            column = find_column(_spi.split('=')[0])
                            value = _spi.split('=')[1].strip()
                            if column in ['category','marked','date']:
                                auto_categories[-1]['change'][column] = value
                            elif column in ['text','posted_date','info']:
                                auto_categories[-1]['search'][column] = value

        
        return categories,auto_categories
    elif _type == 'marked':
        # create an empty dictionary to store DataFrames for each line
        marked = {}
        
        # open the text file for reading
        with open("settings." + _type) as f:
            current_subcategory = None
            
            for line in f:
                if len(line) > 0:
                    if line[0] != '#':
                        _split = line.split(';')
                        mark_name = _split[0].capitalize()
                        start_date = ''
                        end_date = ''
                        value = 0
                        if len(_split) > 1:
                            start_date = _split[1]
                            if len(_split) > 2:
                                end_date = _split[2]
                                if len(_split) > 3:
                                    value = float(_split[3])
                        if mark_name not in marked:
                            marked[mark_name] = {}
                            marked[mark_name]['start'] = [start_date]
                            marked[mark_name]['end'] = [end_date]
                            marked[mark_name]['value'] = [value]
                        else:
                            marked[mark_name]['start'].append(start_date)
                            marked[mark_name]['end'].append(end_date)
                            marked[mark_name]['value'].append(value)
        return marked
    elif _type == 'import' and False:
        # Convert the string representation to a dictionary using ast.literal_eval
        settings = {}
        with open("settings." + _type) as f:
            for line in f:
                if '=' in line:
                    _set = line.strip().split("=")[0]
                    settings[_set] = line.strip().split("=")[1]
        return settings
    elif _type == 'views':
        settings = {}
        filters = [0]*100
        views = [0]*100
        # Read content from the file
        with open("settings." + _type) as f:
            content = f.read()
        exec(content)
        filters_out = []
        for f in filters:
            if f != 0:
                filters_out.append(f)
        settings['filters'] = filters_out
        views_out = []
        for v in views:
            if v != 0:
                views_out.append(v)
        settings['views'] = views_out
        return settings
    else:
        settings = {}
        with open("settings." + _type) as f:
            for line in f:
                if '=' in line:
                    _set = line.strip().split("=")[0]
                    if '[' in line and ']' in line:
                        _val = line.strip().split("=")[1].strip("[]").split(",")
                        settings[_set] = _val
                    else:
                        settings[_set] = line.strip().split("=")[1]
        return settings

def get_newest_csv_file(import_folder):
    # find newest csv file in folder
    list_of_files = glob.glob(os.path.join(import_folder, "*.csv"))
    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f'Importing {latest_file}')
        print('')
        found_csv = True
    else:
        print(f'No csv files found in {import_folder}')
        latest_file = None
    return latest_file

def read_csv_file(file_path, headers, csv_sep, csv_encoding, decimal_sep, thousand_sep):
    # read csv file
    new_data = pd.read_csv(file_path, sep=csv_sep, header=None, names=headers, dtype=str, encoding=csv_encoding)

    # Remove the 'ignore' column if it exists
    if 'ignore' in new_data.columns:
        new_data = new_data.drop('ignore', axis=1) 

    # convert value and balance to float
    new_data["value"] = new_data["value"].str.replace(thousand_sep, "", regex=True).str.replace(decimal_sep, ".", regex=True).astype(float)
    new_data["balance"] = new_data["balance"].str.replace(thousand_sep, "", regex=True).str.replace(decimal_sep, ".", regex=True).astype(float)
    
    return new_data

def add_missing_columns(data):
    add_col = ['id','text','info','notes','account','category_type','category','split_id','marked','date','amount','status']
    for _column in add_col:
        if _column not in data.columns:
            # insert an empty column
            data[_column] = pd.Series(dtype='object')
    for index, row in data.iterrows():
        # Make a timestamp to a pandas datetime object
        dt = pd.to_datetime(datetime.now())

        # insert the datetime object into the 'timestamp' column of the dataframe
        data.at[index, 'imported'] = dt
        data.at[index, 'changed'] = dt
    data[add_col[1:]] = data[add_col[1:]].fillna('')
    return data

def import_data(input_df=None,new=False):
    print('')
    print('')
    import_setting = import_settings('import')
    headers = import_setting["headers"]
    csv_sep = import_setting["csv_sep"]
    decimal_sep = import_setting["decimal_sep"]
    thousand_sep = import_setting["thousand_sep"]
    date_format = import_setting["date_format"]
    import_folder = import_setting.get("import_folder", None)
    csv_encoding = import_setting["csv_encoding"]

    min_col = ['posted_date','text','value','balance']

    mis_col = []
    for col in min_col:
        if col not in headers:
            mis_col.append(col)
    if len(mis_col) > 0:
        print('The following headers in not found in header settings.')
        for _ in mis_col:
            print(_)

    found_csv = False
    # find newest csv file in folder
    if import_folder:
        latest_file = get_newest_csv_file(import_folder)
        if latest_file is not None:
            found_csv = True
    if not found_csv:
        root = tk.Tk()
        root.withdraw()
        latest_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not latest_file:
            return None,None
    new_data = read_csv_file(latest_file, headers, csv_sep, csv_encoding, decimal_sep, thousand_sep)
    new_data['posted_date'] = pd.to_datetime(new_data['posted_date'], format=date_format)
    if 'date' in new_data:
        new_data['date'] = pd.to_datetime(new_data['date'], format=date_format)
    new_data = add_missing_columns(new_data)

    # check if data should be appended to existing dataframe
    match_found = False 
    if input_df is not None:
        df = input_df.copy()
        if len(df) >= 3:
            df_col = [df.columns.get_loc('posted_date'), df.columns.get_loc('value'), df.columns.get_loc('balance')]
            nd_col = [new_data.columns.get_loc('posted_date'), new_data.columns.get_loc('value'), new_data.columns.get_loc('balance')]
            i = -1
            for _account in df['account'].unique():
                df_check_no_split = df.iloc[ df.index[(df['split_id'] < 0) & (df['account'] == _account)].tolist(),:].reset_index(drop=True).copy()
                df_check = df_check_no_split.iloc[-3:,df_col].reset_index(drop=True)
                for i in range(len(new_data) - 3):
                    nd_check = new_data.iloc[i:i+3,nd_col].reset_index(drop=True)
                    if df_check.equals(nd_check):
                        # If a match is found, break
                        match_found = True 
                        break
                if match_found:
                    break
            if match_found:
                new_data = new_data.iloc[i+3:]
                initial_id = df['id'].iloc[-1] + 1
                new_data['id'] = range(initial_id,initial_id + len(new_data))
                new_data['split_id'] = -1
                new_data['amount'] = new_data['value']
                new_data['date'] = new_data['posted_date']
                new_data['account'] = _account

                o_sh = df.shape
                # set id missing
                df = pd.concat([df, new_data.iloc[:]], ignore_index=True)
                n_sh = df.shape
                tprint("Imported data",font="aquaplan")
                print(new_data)              
                print('')
                print('Dataframe is change from')
                print(f'[{o_sh[0]} rows x {o_sh[1]} columns]')
                print('to')
                print(f'[{n_sh[0]} rows x {n_sh[1]} columns]')
                match_found = True
            else:
                tprint("Error importing",font="aquaplan")
                warnings.warn(" Existing dataframe does not match new data.")
                print(' New data for import:')
                print(new_data)
                print('')
                if len(new_data) > 2:
                    print('Create a new account for the data?')
                    import_err_input = input('y/n: ')
                    if import_err_input.lower() == 'y':
                        account_input = input('New account name: ')
                        if not (df['account'].isin([account_input]).any() and len(account_input) > 0):
                            new_data['id'] = range(len(new_data))
                            new_data['split_id'] = -1
                            new_data['amount'] = new_data['value']
                            new_data['date'] = new_data['posted_date']
                            new_data['account'] = account_input
                            if df is None:
                                df = new_data
                            else:
                                df = df.append(new_data, ignore_index=True)
                            match_found = True
                            tprint("New dataframe",font="aquaplan")
                            print(new_data)              
                else:
                    warnings.warn('Imported data needs minimum 3 rows')
    else:
        if len(new_data) > 2:
            account_input = input('New account name: ')
            if len(account_input) > 0:
                new_data['id'] = range(len(new_data))
                df = new_data
                df['split_id'] = -1
                df['amount'] = df['value']
                df['date'] = df['posted_date']
                df['account'] = account_input
                tprint("New dataframe",font="aquaplan")
                print(new_data)              
                match_found = True
        else:
            warnings.warn('Imported data needs minimum 3 rows')
    if df is not None and match_found == True:
        return df
    else:
        return None

def update_view(df,df_fil,v,categories,get_sel=0):
    # Select correct view
    print_col = settings['views'][v["view"]]['columns']
    align_col = settings['views'][v["view"]]['align']
    
    '''
        print_col = vfil["balance_columns"]
        align_col = vfil["balance_align"]
    if v["status"] == "categories":
        print_col = vfil["categories_columns"]
        align_col = vfil["categories_align"]
    '''
    # Set filters based on type
    filters = 0
    # Get filtered dataframe
    df_filtered = filter_dataframe(df, df_fil)
    #Sort
    sort2_col = ''
    sort_dir = 'des'
    sort2_dir = 'des'
    if 'sort' in df_fil and 'column' in df_fil['sort']:
        sort_col = df_fil['sort']['column']
    else:
        sort_col = ''
    if 'sort2' in df_fil and 'column' in df_fil['sort2']:
        sort2_col = df_fil['sort2']['column']
    else:
        sort2_col = ''
    if 'sort' in df_fil and 'direction' in df_fil['sort']:
        sort_dir = df_fil['sort']['direction']
    else:
        sort_dir = ''
    if 'sort2' in df_fil and 'direction' in df_fil['sort2']:
        sort2_dir = df_fil['sort2']['direction']
    else:
        sort2_dir = ''
    if len(sort_col) == 0:
        if 'name' in df_filtered:
            sort_col = 'name'
        else:
            sort_col = 'posted_date'
    if len(sort2_col) == 0:
        sort2_col = 'id'
    if len(sort2_dir) == 0:
        sort2_dir = 'des'
    if sort_dir != 'des':
        sort_asc = False
    else:
        sort_asc = True
    if sort2_dir != 'des':
        sort2_asc = False
    else:
        sort2_asc = True
    if sort_col == 'category':
        sort_col = 'category_sort'
    if sort2_col == 'category':
        sort2_col = 'category_sort'

    df_sorted = df_filtered.sort_values(by=[sort_col,sort2_col], ascending=[sort_asc,sort2_asc])

    pages = m.ceil(len(df_sorted)/int(settings["view_rows"]))
    total_length = len(df_sorted)
    # Dataframe to show
    page = int(v["page"]) 
    if page < 1:
        page = 1
    if page > pages:
        page = pages
    v["page"] = str(page)
    '''
    if page > 1:
      lloc = (1-page)*int(v["pages"])
    else:
      lloc = 1e10
    if page < pages:
        floc = (pages - page - 1) * int(v["pages"]) + (total_length % int(v["pages"]))
    else:
        flic = 0
    '''
    if get_sel != 'all':
        if page > 1:
            df_sorted = df_sorted.iloc[:(1-page)*int(settings["view_rows"]),:].copy()
        if page < pages: 
            df_sorted = df_sorted.iloc[(pages - page - 1) * int(settings['view_rows']) + (total_length % int(settings["view_rows"])):,:]
    
    if get_sel == 0:
        if 'posted_date' in df_sorted:
            df_sorted['posted_date'] = df_sorted['posted_date'].dt.strftime('%d %b %Y')
        if 'date' in df_sorted:
            df_sorted['date'] = df_sorted['date'].dt.strftime('%d %b %Y')
        # Changing category column to name.
        
        if 'category' in settings['views'][v['view']]['columns']:
            for index, row in df_sorted.iterrows():
                category_type = row['category_type']
                category = row['category']
                if len(str(category_type)):
                    if isinstance(category_type,int) or isinstance(category_type,str) or len(category_type.split(',') == 1):
                        if isinstance(category_type,str):
                            category_type = int(category_type[0])
                if len(category) > 0:
                    if isinstance(row["category_type"],str) and len(row["category_type"].split(',')) > 1:
                        unique_types = list(set(row["category_type"].split(',')))
                        if len(unique_types) == 1:
                            category_name = categories[int(unique_types[0])]['name']
                        else:
                            category_name = 'Multiple'
                        # if they are the same as below otherwise Multiple
                    else:
                        category_name = categories[category_type]['name']
                    if category_name == 'Multiple':
                        sub_category_name = ''
                    else:
                        if len(category.split(',')) > 1:
                            sub_category_name = 'Multiple'
                        else:
                            sub_category_name = categories[category_type][category]
                    if len(sub_category_name):
                        _text = f"{category_name} - {sub_category_name}"
                    else:
                        _text = f"{category_name}"
                    #_text = category_name + ' - ' + sub_category_name
                    df_sorted.at[index, 'category'] = _text.strip()
        if v['type'] == 'budget':
            if v['seperate_marked_in_budget']:
                mar_list = autocomplete_marked(marked)
                #mar_dic = {index: value for index, value in enumerate(mar_list)}
                mar_dic = {key: {} for key in mar_list}
                for _m,_item in marked.items():
                    for start,end,value in zip(_item['start'],_item['end'],_item['value']):
                        start_date = pd.to_datetime(start, format='%Y-%m')
                        end_date = pd.to_datetime(end, format='%Y-%m')
                        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
                        for i in range(12):
                            _date = datetime.strptime(str(i+1)+'-'+str(v['year']), '%m-%Y')
                            value_sum = 0
                            if i == 0:
                                count_dates = sum(date_range < _date)
                                value_sum += count_dates*value
                            if _date in date_range:
                                value_sum += value
                            _mon = get_months(i+1)
                            if _mon not in mar_dic[_m]:
                                mar_dic[_m].update({_mon:value_sum})
                            else:
                                new_val = mar_dic[_m][_mon] + value_sum
                                mar_dic[_m][_mon] = new_val
                for _m,_item in marked.items():
                    post_df = dfs_history[current_history_index].copy()
                    max_date = datetime.strptime(str(v['year']-1)+'-12-31', '%Y-%m-%d')
                    marked_filter = {"key": "", "type": "posts","marked":{"equal":[_m]},"date":{"max":max_date}}
                    post_filtered = filter_dataframe(post_df, marked_filter)
                    past_years_used = post_filtered['amount'].sum()
                    mar_dic[_m]['jan'] += past_years_used
                #new_rows = pd.DataFrame.from_dict({(i, 'name'): j for i, j in mar_dic.items()}, orient='index')
                # Convert the dictionary to a DataFrame
                new_rows = pd.DataFrame.from_dict(mar_dic, orient='index')
                new_rows['category'] = 'Marked'

                # Reset the index and rename the 'index' column to 'name'
                new_rows = new_rows.reset_index().rename(columns={'index': 'name'})
                
                # Create a list of datetimes with a monthly frequency
                #new_rows = pd.DataFrame({'name': name_values})
                #df_sorted = pd.concat([new_rows, df_sorted], ignore_index=True)
                df_sorted = pd.concat([df_sorted,new_rows], ignore_index=True)
            _month_col = get_months()
            df_sorted['tot'] = df_sorted[_month_col].sum(axis=1)
        if len(df_sorted) > 0:
            float_format = lambda x: f"{x:,.2f}"
            for c in ['amount','value','balance']:
                if c in df_sorted.columns:
                    df_sorted[c] = df_sorted[c].apply(float_format)
            float_format = lambda x: f"{x:,.0f}"
            for c in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','tot']:
                if c in df_sorted.columns:
                    df_sorted[c] = df_sorted[c].apply(float_format)

        df_sorted.loc[:,'select'] = range(len(df_sorted), 0, -1)
        table = pt(list(['select'] + print_col))
        table._max_width = {col: 32 for col in print_col}
        for col, align in zip(print_col, align_col):
            table.align[col] = align
        for index,row in df_sorted[['select'] + print_col].iterrows():
            table.add_row(row)
        print('')
        print_header = settings['views'][v['view']]['name'].format(year=v['year'])
        tprint(print_header,font="aquaplan")
        print(table)
        print("Page " + str(page) + " of " + str(pages))
        return v
    else:
        # Get and return row number
        if get_sel == 'all':
            return df_sorted['id']
        else:
            _list = convert_string_to_list(get_sel)
            n = len(df_sorted)  # number of rows in the DataFrame
            sel_list = [n - i for i in _list]  # convert to zero-based indices
            return df_sorted.iloc[sel_list]['id']

def filter_dataframe(df, df_filter,get_ids=False):
    """
    Filter rows in a dataframe based on filter values specified for each column.

    Args:
        df (pd.DataFrame): The input dataframe to filter.
        df_filter (Dict[str, Any]): A dictionary where keys are column names and values are filter values.

    Returns:
        pd.DataFrame: A filtered dataframe based on filter values.
    """
    filtered_df = df.copy()
    for column_name, column_filter in df_filter.items():
        if column_name not in df.columns:
            continue  # Ignore filters for non-existing columns
        column_type = df[column_name].dtype
        if column_type == 'object':  # string column
            contains = column_filter.get('contains', None)
            equal = column_filter.get('equal', None)
            empty = column_filter.get('empty', False)
            not_empty = column_filter.get('not_empty', True)

            # Handle multiple 'contains' values
            if contains is not None:
                filtered_dfs = []
                if not not_empty:
                    filtered = filtered_df[filtered_df[column_name] == '']
                    filtered_dfs.append(filtered)
                else:
                    if empty:
                        filtered = filtered_df[filtered_df[column_name] == '']
                        filtered_dfs.append(filtered)
                    if len(contains) > 0:
                        if isinstance(contains, list):
                            contains_condition = filtered_df[column_name].str.contains('|'.join(contains), case=False, na=False)
                        else:
                            contains_condition = filtered_df[column_name].str.contains(contains, case=False, na=False)
                        filtered = filtered_df.loc[contains_condition]
                        filtered_dfs.append(filtered)
                filtered_df = pd.concat(filtered_dfs,ignore_index=False)
            elif equal is not None or not not_empty:
                if column_name == 'category':
                    filtered_dfs = []
                    if not not_empty:
                        filtered = filtered_df[filtered_df['category'] == '']
                        filtered_dfs.append(filtered)
                    else:
                        cat_index = []
                        if len(equal) > 0:
                            filtered_dfs = []
                            for eq_cat in equal:
                                filtered = filtered_df[(filtered_df['category_type'] == int(eq_cat[:-1])) & (filtered_df['category'] == eq_cat[-1])]
                                filtered_dfs.append(filtered)
                            if empty:
                                filtered = filtered_df[filtered_df['category'] == '']
                                filtered_dfs.append(filtered)
                    filtered_df = pd.concat(filtered_dfs,ignore_index=False)
                elif column_name == 'marked':
                    filtered_dfs = []
                    if not not_empty:
                        filtered = filtered_df[filtered_df['marked'] == '']
                        filtered_dfs.append(filtered)
                    else:
                        for eq_mar in equal:
                            filtered = filtered_df[filtered_df['marked'] == eq_mar]
                            filtered_dfs.append(filtered)
                        if empty:
                            filtered = filtered_df[filtered_df['marked'] == '']
                            filtered_dfs.append(filtered)
                    filtered_df = pd.concat(filtered_dfs,ignore_index=False)
                else:
                    filtered_dfs = []
                    if not not_empty:
                        filtered = filtered_df[filtered_df[column_name] == '']
                        filtered_dfs.append(filtered)
                    else:
                        if empty:
                            filtered = filtered_df[filtered_df[column_name] == '']
                            filtered_dfs.append(filtered)
                        filtered_df = filtered_df.loc[filtered_df[column_name].str.equal(equal)]
                        filtered_dfs.append(filtered)
                    filtered_df = pd.concat(filtered_dfs,ignore_index=False)
        elif column_type == 'datetime64[ns]':  # datetime column
            lower_bound = column_filter.get('min', None)
            upper_bound = column_filter.get('max', None)
            if lower_bound is not None:
                filtered_df = filtered_df.loc[filtered_df[column_name] >= lower_bound]
            if upper_bound is not None:
                filtered_df = filtered_df.loc[filtered_df[column_name] <= upper_bound]
        else:  # numeric column
            lower_bound = column_filter.get('min', None)
            upper_bound = column_filter.get('max', None)
            if lower_bound is not None:
                filtered_df = filtered_df.loc[filtered_df[column_name] >= float(lower_bound)]
            if upper_bound is not None:
                filtered_df = filtered_df.loc[filtered_df[column_name] <= float(upper_bound)]
    if not get_ids: 
        return filtered_df
    else:
        #not used
        return filtered_df['id']

def find_column(input_string,read_only=True):
    # Define a dictionary of predefined words with their corresponding 3-letter keys
    if read_only:
        words = {
            'id': 'id',
            'tex': 'text',
            'inf': 'info',
            'use': 'user_notes',
            'cat': 'category_id',
            'acc': 'account',
            'dat': 'date',
            'pos': 'posted_date',
            'val': 'value',
            'amo': 'amount',
            'bal': 'balance',
            'imp': 'imported',
            'cha': 'changed',
            'cat': 'category',
            'spl': 'split_id',
            'sta': 'status',
            'mar': 'marked',
            'not': 'notes'
        }
    else:
        words = {
            'not': 'notes',
            'cat': 'category',
            'dat': 'date',
            'sta': 'status',
            'mar': 'marked'
        }
    # Get the first 3 characters of the input string
    if len(input_string) > 2:
        key = input_string[:3]
    else:
        key = input_string

    # Check if the key is in the dictionary
    if key in words:
        # Return the full word corresponding to the key
        return words[key]
    else:
        # If the key is not in the dictionary, return None
        return None

def change_dataframe_rows(df,v , column_to_change, dataframe_ids, new_value):
    """
    Changes one or multiple rows of a dataframe by setting the specified column to a new value.

    Parameters:
        df (pandas.DataFrame): The original dataframe.
        column_to_change (str): The name of the column to change.
        dataframe_ids (pandas.DataFrame): A filtered version of the original dataframe that only contains the IDs of the rows to change.
        new_value: The new value to assign to the selected rows in the specified column.

    Returns:
        pandas.DataFrame: The modified dataframe with the selected rows changed.
    """

    # Check if the specified column exists in the DataFrame
    if column_to_change not in df.columns:
        raise ValueError(f"Column '{column_to_change}' not found in DataFrame.")

    # Check if the 'id' column exists in both dataframes
    if 'id' not in df.columns:
        raise ValueError("Column 'id' not found in DataFrame.")
    
    # Get the IDs of the rows to be changed
    ids_to_change = dataframe_ids.unique()

    # Update the original DataFrame with the modified rows
    df.loc[df['id'].isin(ids_to_change),column_to_change] = new_value
    df.loc[df['id'].isin(ids_to_change),'changed'] = pd.to_datetime(datetime.now())

    return df 

def auto_change_dataframe(df, df_filt,cur_view, auto, input_ids):
    mod_df = df.copy()
    for auto_value in auto: 
        mod_filt = df_filt.copy()
        for _c,_v in auto_value['search'].items():
            mod_filt = change_filter(mod_filt,_c,_v)
        filter_ids = update_view(df, mod_filt, cur_view,None,'all')
        combined_ids = pd.concat([input_ids, filter_ids],ignore_index=True)
        change_ids = combined_ids[combined_ids.duplicated()]
        for _c,_v in auto_value['change'].items():
           _value = convert_value_to_data_type(df,cur_view,_c,_v)
           if _value is not None:
               if _c == 'category':
                   if _v != '':
                       _v = autocomplete_category(categories,_v)
                   else:
                       _v = ['','']
                   if _v is not None:
                       mod_df = change_dataframe_rows(mod_df,cur_view,_c,change_ids,_v[1])
                       mod_df = change_dataframe_rows(mod_df,cur_view,'category_type',change_ids,_v[0])
               elif _c == 'marked':
                   if _v != '':
                       _v = autocomplete_marked(marked,_v)[0]
                   if _v is not None:
                       mod_df = change_dataframe_rows(mod_df,cur_view,_c,change_ids,_v)
               else:
                   mod_df = change_dataframe_rows(mod_df,view,_c,change_ids,_v)
        
        #Change all change_ids in mod_df
    return mod_df
    ''' 
        #Keep using df for getting ids.
    new_dfs = auto_change_dataframe(new_dfs,df_filters, view,auto_categories,change_ids)
    change_ids = update_view(new_dfs, df_filt, cur_view, view_filt,None,'all')
    # Check if the specified column exists in the DataFrame
    if column_to_change not in df.columns:
        raise ValueError(f"Column '{column_to_change}' not found in DataFrame.")

    # Check if the 'id' column exists in both dataframes
    if 'id' not in df.columns:
        raise ValueError("Column 'id' not found in DataFrame.")
    
    # Get the IDs of the rows to be changed
    ids_to_change = dataframe_ids.unique()

    # Update the original DataFrame with the modified rows
    df.loc[df['id'].isin(ids_to_change),column_to_change] = new_value

    return df 
    '''
def split_dataframe_rows(df, dataframe_ids, new_value, new_category, new_mark, new_note):
    """
    Splits selected rows of a dataframe by subtracting a new value from the existing value.

    Parameters:
        dfs (list of pandas.DataFrame): The original dataframes.
        v (dict): A dictionary containing information about the operation.
        column_to_split (str): The name of the column to split.
        dataframe_ids (pandas.DataFrame): A filtered version of the original dataframe that only contains the IDs of the rows to split.
        new_value: The value to subtract from the existing values in the specified column.
        new_mark: The new value to assign to the 'marked' column in the new rows.
        new_comment: The new value to assign to the 'comment' column in the new rows.

    Returns:
        list of pandas.DataFrame: The modified dataframes with the selected rows split.
    """
    # Get the IDs of the rows to be split
    ids_to_split = dataframe_ids.unique()

    # Get the maximum existing ID value
    max_id = df['id'].max()

    # Create new rows with the updated values and unique IDs
    #new_rows = df.copy()
    new_rows = df.loc[df['id'].isin(ids_to_split)].copy()
    new_ids = range(max_id + 1, max_id + 1 + len(new_rows))
    new_rows['split_id'] = new_rows.apply(lambda row: row['split_id'] if row['split_id'] >= 0 else row['id'], axis=1)
    new_rows['id'] = new_ids
    if new_mark is not None:
        new_rows['marked'] = new_mark
    else:
        new_rows['marked'] = ''
    if new_category != '':
        new_rows['category_type'] = new_category[0]
        new_rows['category'] = new_category[1]
    new_rows['notes'] = new_note
    dt = pd.to_datetime(datetime.now())
    new_rows['changed'] = dt
    new_rows['value'] = 0

    # Subtract the new value from the specified column
    for _id,_nid in zip(ids_to_split,new_ids):
        if not isinstance(new_value,float):
            _new_value = df.loc[df['id']==_id,'amount'] - float(new_value[1:])
        else: _new_value = new_value
        df.loc[df['id']==_id,'amount'] -= _new_value
        new_rows.loc[new_rows['id']==_nid,'amount'] = _new_value

    # Concatenate the original DataFrame and the new rows
    df = pd.concat([df, new_rows], ignore_index=True)

    return df

def delete_split_dataframe_rows(df, dataframe_ids):
    """
    Splits selected rows of a dataframe by subtracting a new value from the existing value.

    Parameters:
        dfs (list of pandas.DataFrame): The original dataframes.
        v (dict): A dictionary containing information about the operation.
        column_to_split (str): The name of the column to split.
        dataframe_ids (pandas.DataFrame): A filtered version of the original dataframe that only contains the IDs of the rows to split.
        new_value: The value to subtract from the existing values in the specified column.
        new_mark: The new value to assign to the 'marked' column in the new rows.
        new_comment: The new value to assign to the 'comment' column in the new rows.

    Returns:
        list of pandas.DataFrame: The modified dataframes with the selected rows split.
    """
    # Get the IDs of the rows to be split
    ids_to_delete = dataframe_ids.unique()

    for index, row in df[((df['id'].isin(ids_to_delete)) & (df['split_id'] >= 0))].iterrows():
        df_id = row['split_id']
        value_to_add = row['amount']
        
        # Find rows where account_id matches and split is 'n'
        row_to_update = df[df['id'] == df_id]
        
        # Update the 'value' column in the selected rows
        df.loc[row_to_update.index, 'amount'] += value_to_add
    
    df = df[~((df['id'].isin(ids_to_delete)) & (df['split_id'] >= 0))]
    
    return df

def convert_string_to_list(string):
    """
    Converts a string in the format of '5,6,8-12' to a list of integers that includes all numbers in between,
    removing duplicates.

    Parameters:
        string (str): The string to convert.

    Returns:
        list: A list of integers.
    """
    result = []

    for part in string.split(','):
        if '-' in part:
            start, end = part.split('-')
            result.extend(range(int(start), int(end)+1))
        else:
            result.append(int(part))

    return list(set(result))

def convert_value_to_data_type(dfs,v , column, value):
    """
    Converts a value to the same data type as the values in a specified column of a DataFrame.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the column.
        column (str): The name of the column.
        value (str): The value to convert.

    Returns:
        object: The converted value with the same data type as the values in the specified column.
    """
    if value == '':
        return ''
    data_type = dfs[column].dtype

    try:
        if data_type == 'int64' or data_type == 'float64':
            return data_type.type(value)
        elif data_type == 'bool':
            return bool(value)
        elif data_type == 'datetime64[ns]':
            if len(value.split('/')) == 2:
                return pd.to_datetime(str(datetime.now().year) + '/' + value)
            else:
                return pd.to_datetime(value)
        elif data_type == 'timedelta64[ns]':
            return pd.to_timedelta(value)
        else:
            return str(value)
    except ValueError:
        return None

def str_to_datetime(date_string):
    i_settings = import_settings('import')
    if isinstance(date_string, int) or '-' not in date_string:
        # if the input is a negative integer, subtract that number of days from the current date
        dt = datetime.now() - timedelta(days=abs(int(date_string)))
    else:
        try:
            # try to parse the date string using the specified format
            dt = datetime.strptime(date_string, i_settings['date_format'])
        except ValueError:
            # if that fails, try to parse the date string as yyyy-mm-dd format
            dt = datetime.strptime(date_string, '%Y-%m-%d')
    return dt

def autocomplete_category(categories, string_bit, output='default'):
    if string_bit != '':

        if output == 'comp_list':
            result = []
        # Check if input is in the format of {category}{subcategory}
        count = 0 
        if len(string_bit) > 1 or string_bit.isdigit():
            if string_bit.isdigit():
                category = string_bit
                subcategory = ''
            else:
                category = string_bit[:-1]
                subcategory = string_bit[-1]
            if category.isdigit():
                category = int(category)
                if category in categories:
                    if subcategory in categories[category]:
                        if output == 'comp_list':
                            count += 1
                            result.append(str(category) + subcategory)
                        elif output == 'default':
                            return [category, subcategory]
                    elif subcategory == '':
                        if output == 'comp_list':
                            for sub, item in categories[category].items():
                                if sub != 'name':
                                    count += 1
                                    result.append(str(category) + sub)
                elif category == 0:
                    for cat, item in categories.items():
                        for sub, item in categories[cat].items():
                            if sub != 'name':
                                count += 1
                                result.append(str(cat) + sub)
        # Check for matches in category names and subcategory names
        if not string_bit.isdigit():
            for category, subcategories in categories.items():
                for subcategory, name in subcategories.items():
                    if string_bit.lower() in name.lower():
                        if subcategory != 'name': 
                            count += 1
                            if output == 'comp_list':
                                result.append(str(category) + subcategory)
                            elif output == 'default':
                                result = [category, subcategory]
                        else:
                            if output == 'comp_list':
                                for subcategory, name in subcategories.items():
                                    if subcategory != 'name': 
                                        count += 1
                                        result.append(str(category) + subcategory)
        if count == 0:
            print('No category match found for ' + string_bit)
        elif count == 1:
            return result
        elif output == 'comp_list':
            return result
        else:
            print(f'{count} matches found for {string_bit}')
        for category, subcategories in categories.items():
            for subcategory, name in subcategories.items():
                if string_bit.lower() in name.lower():
                    A = 1
                    #print(f'{category}{subcategory} {category['name'}} {subcategory[
        
        # Return None if no match is found
    return None

def autocomplete_marked(marked, string_bit='',output='default'):
    mark_list = []
    if string_bit != '':
        for mark, name in marked.items():
            if string_bit.lower() == mark[:min(len(string_bit),len(mark))].lower():
                mark_list.append(mark)
        if len(mark_list) == 0:
            print('No marked options found for ' + string_bit)
        elif output == 'comp_list' or len(mark_list) == 1:
            return mark_list
        else:
            print(f'{len(mark_list)} matches found for {string_bit}')
    else:
        for mark, name in marked.items():
            if mark not in mark_list:
                mark_list.append(mark)
        return mark_list
    # Return None if no match is found
    return None

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            name, start_date, end_date, value = line.strip().split(';')
            start_month, start_year = map(int, start_date.split('-'))
            end_month, end_year = map(int, end_date.split('-'))
            value = int(value)
            start = datetime(start_year, start_month, 1)
            end = datetime(end_year, end_month, 1)
            data.append((name, start, end, value))
    df = pd.DataFrame(data, columns=['Name', 'Start Date', 'End Date', 'Value'])
    return df

def extract_info(df, input_type, input1=None, input2=None):
    if input_type == 'get_mark':
        if input1:
            return df[df['Name'].str.contains(input1, case=False)]
        else:
            return df['Name'].unique()
    elif input_type == 'get_value':
        if input1:
            subset = df[df['Name'].str.contains(input1, case=False)]
        else:
            subset = df.copy()
        if input2:
            date = datetime.strptime(input2, '%m-%Y')
        else:
            date = datetime.now().replace(day=1)
        subset['Effective Months'] = subset.apply(lambda row: min((row['End Date'] - date).days // 30, (date - row['Start Date']).days // 30 + 1), axis=1)
        if input1:
            return subset['Value'].sum() * subset['Effective Months'].sum()
        else:
            return subset.groupby('Name')['Value', 'Effective Months'].sum()

def change_filter(input_filters, input_column,input_value):
     
    if input_column is not None:
        if len(input_value) > 0 and input_value[0] in '<>=':
            if input_column in ['posted_date','date','changed','imported']:
                if input_value[1] == '-':
                    _value = parse_duration_string(input_value[1:])
                else: 
                    _value = str_to_datetime(input_value[1:])
            else:
                _value = float(input_value[1:])
        else:
            _value = input_value.strip('\'')
        if input_value[0] == '>':
            input_filters.setdefault(input_column, {})['min'] = _value
        elif input_value[0] == '<':
            input_filters.setdefault(input_column, {})['max'] = _value
        elif input_value[0] == '=':
            input_filters.setdefault(input_column, {})['min'] = _value
            input_filters.setdefault(input_column, {})['max'] = _value
        elif _value == '/oe': #What is this?
            input_filters.setdefault(input_column, {})['empty'] = True
            input_filters.setdefault(input_column, {})['not_empty'] = False
        elif _value == '/he':
            input_filters.setdefault(input_column, {})['empty'] = False
        elif _value == '/se':
            input_filters.setdefault(input_column, {})['empty'] = True
        else:
            # Add something for category view and category to filter based on category ids instead. Then values should be from ids. Not here but in filter.
    
            if input_column == 'category':
                cat_ids = _value.split(',')
                cat_values = []
                for ci in cat_ids:
                    if ci.isdigit():
                        for key in categories[int(ci)]:
                            if key != 'name':
                                cat_values.append(ci+key)
                    else:
                        cat_value_list = autocomplete_category(categories,ci,output='comp_list')
                        if cat_value_list is not None:
                            for cvl in cat_value_list:
                                cat_values.append(cvl)
                if len(cat_values) > 0:
                    input_filters.setdefault(input_column, {})['equal'] = cat_values
                    input_filters.setdefault(input_column, {})['not_empty'] = True
            elif input_column == 'marked':
                mar_values = []
                mar_ids = _value.split(',')
                for mi in mar_ids:
                    mar_value_list = autocomplete_marked(marked,mi,output='comp_list')
                    if mar_value_list != None:
                        for ma in mar_value_list:
                            if ma not in mar_values:
                                mar_values.append(ma)
                    else:
                        mar_values.append(marked)
                if len(mar_values) > 0:
                    input_filters.setdefault(input_column, {})['equal'] = mar_values
                    input_filters.setdefault(input_column, {})['not_empty'] = True
            else:
                input_filters.setdefault(input_column, {})['contains'] = _value.split(',')
    return input_filters
import re
from datetime import timedelta

def parse_duration_string(duration_input):
    # Define regular expressions for parsing different components
    year_pattern = re.compile(r'(\d+)yea')
    month_pattern = re.compile(r'(\d+)mon')
    day_pattern = re.compile(r'(\d+)day')
    hour_pattern = re.compile(r'(\d+)hou')
    min_pattern = re.compile(r'(\d+)min')
    sec_pattern = re.compile(r'(\d+)sec')
    
    duration_string = duration_input[1:] 
    # Initialize timedelta with default values
    duration = timedelta()

    # Parse each component and add it to the timedelta
    match = year_pattern.search(duration_string)
    if match:
        duration += timedelta(days=int(match.group(1)) * 365)

    match = month_pattern.search(duration_string)
    if match:
        duration += timedelta(days=int(match.group(1)) * 30)

    match = day_pattern.search(duration_string)
    if match:
        duration += timedelta(days=int(match.group(1)))

    match = hour_pattern.search(duration_string)
    if match:
        duration += timedelta(hours=int(match.group(1)))

    match = min_pattern.search(duration_string)
    if match:
        duration += timedelta(minutes=int(match.group(1)))

    match = sec_pattern.search(duration_string)
    if match:
        duration += timedelta(seconds=int(match.group(1)))

    # Get current datetime
    current_datetime = datetime.now()

    # Subtract the duration from the current datetime
    result_datetime = current_datetime - duration

    return result_datetime

def convert_delta_filter_time():
    for i in range(len(settings['filters'])):
        for column_name, column_filter in settings['filters'][i].items():
            if 'date' in column_name or 'stamp' in column_name:
                for bound, time_val in column_filter.items():
                    if '-' == time_val[0]:
                        new_val = parse_duration_string(time_val)
                        settings['filters'][i][column_name][bound] = new_val
def get_months(_i='all'):
    _all = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    if _i == 'all':
        return _all
    elif isinstance(_i,int):
        return _all[_i-1]
    elif isinstance(_i,str):
        if _i.isdigit():
            return _all[int(_i)-1]
def read_budget_files(folder_path='./'):
    # Initialize an empty dictionary to store DataFrames
    budgets_dfs = {}
    month_columns = get_months()

    # List all files in the specified folder
    files = [f for f in os.listdir(folder_path) if f.lower().startswith('budget') and f.endswith('.csv')]

    # Iterate through each file and read it into a DataFrame
    for file in files:
        file_path = os.path.join(folder_path, file)

        # Extract the year from the filename (assuming it's in the format "budget_YYYY.csv")
        year = int(file.lower().split('_')[1].split('.')[0])

        # Read the CSV file into a list of dictionaries
        dfs = []
        _id = -1
        with open(file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)

            # Iterate through each row and append it to the list of dictionaries
            for row in csv_reader:
                all_category = '' 
                all_type = ''
                all_type_list = []
                all_cat_list = []
                cat_sort = ''
                if len(row) and row[0][0] not in '#':
                    for cat in row[1].split(';'):
                        cat_value_list = autocomplete_category(categories,cat,output='comp_list')
                        if not cat_value_list == None:
                            for cat2 in cat_value_list:
                                _t = int(cat2[:len(cat2)-1])
                                _c = cat[-1]
                                all_type += cat2[:len(cat2)-1]+','
                                all_type_list.append(cat2[:len(cat2)-1])
                                all_cat_list.append(cat2[-1])
                                all_category += cat2[-1]+','
                                cat_sort += "{:02d}".format(_t) + _c + ','
                    unique_types = list(set(all_type_list))
                    unique_cat = list(set(all_cat_list))
                    if len(unique_types) > 1:
                        cat_sort = '!'+cat_sort
                    elif len(unique_cat) > 1:
                        ins_pos = len(str(unique_types[0]))
                        cat_sort = cat_sort[:2] + '0' + cat_sort[2:]
                    if len(all_type)>1:
                        all_type = all_type[:-1]
                    if len(all_category)>1:
                        all_category = all_category[:-1]
                    _id += 1
                    row_data = {
                        'year': year,
                        'name': row[0],
                        'categories': row[1],
                        'category_type': all_type,
                        'category': all_category,
                        'category_sort': cat_sort,
                        'tot': 0,
                        'id': _id,
                        # Add other columns as needed
                    }
                    row_data.update({f'{month}': 0.0 for month in month_columns})
                    if len(row) == 14:
                        row_data.update({f'{month}': float(value) for month, value in zip(month_columns, row[2:])})
                    elif len(row) == 4:
                        for mon, val in zip(row[2].split(';'), row[3].split(';')):
                            if ':' in mon:
                                if mon[0] == ':':
                                    mon1 = 1
                                else:
                                    mon1 = int(mon.split(':')[0])
                                if mon[-1] == ':':
                                    mon2 = 12
                                else:
                                    mon2 = int(mon.split(':')[1])
                                for m in range(mon1-1, mon2):
                                    row_data[f'{month_columns[m]}'] = float(val)
                            else:
                                row_data[f'{month_columns[int(mon)-1]}'] = float(val)
                    #tot_val = 0
                    #for month in month_columns:
                    #    tot_val += row_data[f'{month}'] 
                    #row_data['tot'] = tot_val
                    dfs.append(row_data)
            for row in csv_reader:
                negative numbers should get remaining categories

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(dfs)

        budgets_dfs[year] = df

    return budgets_dfs

def group_and_sum_rows(df, group_level=1):
    if group_level == 1:
        group_columns = ['category_type', 'category']
    elif group_level == 2:
        group_columns = ['category_type']
    else:
        raise ValueError("group_level should be 1 or 2.")

    # Create a mask to identify rows where 'category_sort' starts with 0
    mask_starts_with_zero = df['category_sort'].astype(str).str.startswith('!')

    # Group by specified columns and sum the numeric columns for rows not starting with 0
    grouped_df = df[~mask_starts_with_zero].groupby(group_columns, as_index=False).agg({
        'year': 'first',
        'categories': lambda x: ', '.join(x.astype(str)),
        'category_sort': lambda x: ', '.join(x.astype(str)),
        'jan': 'sum', 'feb': 'sum', 'mar': 'sum', 'apr': 'sum', 'may': 'sum',
        'jun': 'sum', 'jul': 'sum', 'aug': 'sum', 'sep': 'sum', 'oct': 'sum',
        'nov': 'sum', 'dec': 'sum', 'tot': 'sum',
        'name': lambda x: ', '.join(x),
        'id': lambda x: ', '.join(x.astype(str))
    })

    # Merge the grouped DataFrame with the original DataFrame to keep all rows
    #result_df = pd.merge(df, grouped_df, on=group_columns, how='left')

    # For rows where 'category_sort' starts with 0, copy the original values without summing
    #grouped_df.loc[mask_starts_with_zero, ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec', 'tot']] = \
    #grouped_df.loc[mask_starts_with_zero, ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'okt', 'nov', 'dec', 'tot']].values
    #grouped_df.loc[mask_starts_with_zero, 'name'] = df.loc[mask_starts_with_zero, 'name'].values

    # Rearrange the columns
    #grouped_df = grouped_df[['id', 'year_y', 'name', 'categories_y', 'category_type',
    #                       'category_sort_y', 'jan', 'feb', 'mar', 'apr', 'may',
    #                       'jun', 'jul', 'aug', 'sep', 'okt', 'nov',
    #                       'dec', 'tot']]

    # Rename the summed columns
    #$grouped_df.rename(columns={'year_y': 'year', 'categories_y': 'categories', 'category_sort_y': 'category_sort',
    #$                          'jan': 'jan', 'feb': 'feb', 'mar': 'mar', 'apr': 'apr', 'may': 'may',
    #$                          'jun': 'jun', 'jul': 'jul', 'aug': 'aug', 'sep': 'sep', 'okt': 'okt',
    #$                          'nov': 'nov', 'dec': 'dec', 'tot': 'tot'}, inplace=True)
    grouped_df.rename(columns={'year': 'year', 'categories_y': 'categories', 'category_sort_y': 'category_sort',
                              'jan': 'jan', 'feb': 'feb', 'mar': 'mar', 'apr': 'apr', 'may': 'may',
                              'jun': 'jun', 'jul': 'jul', 'aug': 'aug', 'sep': 'sep', 'oct': 'oct',
                              'nov': 'nov', 'dec': 'dec', 'tot': 'tot'}, inplace=True)
    result_df = pd.concat([grouped_df,df[mask_starts_with_zero]], ignore_index=True)
    

    return result_df
                
def budget_status(df,post_df):
    month_mapping = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
    }
    filters_ini = {"key": "", "type": "posts"}
    # Loop over each row in the DataFrame and update budget values
    for index, row in df.iterrows():
        year = row['year']
        for month in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']:
            #Current:{'key': 'cat', 'type': 'posts', 'sort': {'column': 'date', 'direction': 'des'},
            #         'sort2': {'column': 'id', 'direction': 'des'}, 'status': {'contains': ['ok']},
            #         'date': {'min': datetime.datetime(2023, 1, 5, 0, 0), 'max': datetime.datetime(2023, 1, 12, 0, 0)},
            #         'category': {'equal': ['4b', '3d'], 'not_empty': True}
            #         }


            _mon = month_mapping.get(month)
            filt = filters_ini.copy()
            date1 = datetime(year,_mon,1)
            _, last_day = calendar.monthrange(year, _mon)
            date2 = datetime(year,_mon,last_day)
            filt['date'] = {'min': date1,'max':date2}
            cat = []
            for _typ,_cat in zip(row['category_type'].split(','),row['category'].split(',')):
                cat.append(str(_typ) + _cat)
            filt['category'] = {'equal': cat, 'not_empty':True}
            filter_post_df = filter_dataframe(post_df, filt)
            post_sum = filter_post_df['amount'].sum()
            df.at[index, month] = post_sum - df.at[index, month]

    return df 

def print_help(commands,view):
    if commands == 'basic':
        print('Switch views:')
        print('i=import posts - q=quit')
        print('key=to switch for predefined view and filter')
        print('u=undo latest command - r=redo latest command')
        print('l=load previous data - w=write data')
        print('h=help, t=update, d=post detail')
        print()
        print('Change filter or values')
        print('f=filter posts - c=change column value - s=split a row')
        print('enter f,c or s for detailts on each function')
        print()
        print('Page options:')
        print('pn   Go to next page')
        print('pp   Go to previous page')
        print('p##  Go to page number')
    elif commands == 'start':
        print('Switch views:')
        print('i=import posts - q=quit')
        print('l=load previous data')
        print('h=help')
    elif commands == 'detail':
        print('Get details of a post')
        print()
        print('Detail:')
        print('d ##')
        print()
        print('Where ## needs to be:')
        print('    A single select number')
    elif commands == 'filter':
        print('Filter and sorting')
        print()
        print('Filter:')
        print('f column filter_option><=value or text for contains')
        print()
        print('Where column can be:')
        print('    The full name of the column to filter')
        print('    The first 3 characters of the column to filter')
        print()
        print('Where filter_option can be:')
        print('    >value, <value or =value to filter based on value')
        print('    /he to hide empty colums')
        print('    /se to show empty colums')
        print('    /oe to show only empty colums')
        print('    string to filter based in a the string is in the column')
        print()
        print('For filtering on time, use either date string(2023-11-15) or relative date from today(-1year2month3day4hour5min6sec)')
        print()
        print('Other filter options:')
        print('f reset  - reset all filter')
        print('f del column  - delete filters on a column')
        print('f column  - delete filters on a column')
        print()
        print('To filter for more than 1 option, use comma to seperate')
        print('f text apple,banana  - to filter apple and banana texts')
        print()
        print()
        print('Sorting:')
        print('f sort column')
        print('f sort dir')
        print('f sort2 column')
        print('f sort2 dir')
        print()
        print('Where dir can be:')
        print('    asc for ascending direction')
        print('    des for descending direction')
    elif commands == 'change':
        print('Change column options:')
        print('c ## column val')
        print()
        print(' To delete a column value, use cd:')
        print('cd ## column')
        print()
        print('Where column can be:')
        print('    The full name of the column to change')
        print('    The first 3 characters of the column to change')
        print()
        print('Where ## can be:')
        print('    A single select number')
        print('    Multiple select numbers eg. 1,6,7')
        print('    Range of select numbers eg. 5-9')
        print('    all for all rows in all pages in current filter')
    elif commands == 'split':
        print('Split category options:')
        print('s ## new_value new_category new_mark new_note')
        print()
        print('Where ## can be:')
        print('    A single select number')
        print('    Multiple select numbers eg. 1,6,7')
        print('    Range of select numbers eg. 5-9')
        print('    all for all rows in all pages in current filter')
        print()
        print('Where new_value is the value of new category.')
        print('  Old value is auto calculated')
        print('  Using -- or -+ in front of the number, means the second sign and value is the old value and new value is auto calculated')
        print()
        print('Where category is given in same way as in change function. If empty by only writing s ## new_value or')
        print('s ## new_value  mark_option (double space) the category will be identical to the original')
        print()
        print('Where new_mark is optional, but given in same way as in change function')
        print()
        print('Where new_note is marked by y, to indicate a new note. Given during the split.')
        print()
        
if __name__ == '__main__':
    # Enable history navigation
    readline.parse_and_bind("tab: complete")
    readline.set_history_length(1000)
    settings = import_settings('general')
    view_setting = import_settings('views')
    settings['filters'] = view_setting['filters']
    settings['views'] = view_setting['views']
    categories,auto_categories = import_settings('categories')
    marked = import_settings('marked')
    budgets = read_budget_files()

    convert_delta_filter_time()
    
    '''
    view_filter = {}
    view_filter["rows"] = settings["view_rows"]
    view_filter["balance_sort"] = settings["view_balance_sort"]
    view_filter["balance_columns"] = settings["view_balance_columns"]
    view_filter["balance_align"] = settings["view_balance_align"]
    view_filter["categories_sort"] = settings["view_categories_sort"]
    view_filter["categories_columns"] = settings["view_categories_columns"]
    view_filter["categories_align"] = settings["view_categories_align"]
    #same filters
    # date,text,value,timestamp
    '''
    view = {}
    view["view"] = 0
    #default budget_view
    view["budget_view"] = ['budget','all',0] #budget/status, 1/2/3 for all/categories/types 0/1 seperate marked
    #budget_filter
    view["budget_filter"] = {}
    view["filter"] = 0
    view["type"] = settings['filters'][view["filter"]]["type"]
    view["seperate_marked_in_budget"] = bool(settings.get("seperate_marked_in_budget",True))
    #import_folder = import_settings.get("import_folder", None)
    view["page"] = '1'
    view["year"] = 0
    # Reset filter to settings filter when reset
    # Define filter using a view key
    #view["pages"] = settings["view_rows"]
    
    #df_default_filter = {'balance_sort':{'column':view_filter["balance_sort"],'direction':'des'},'categories_sort':{'column':view_filter["categories_sort"],'direction':'des'}}
    df_filters = settings['filters'][view["filter"]].copy()
    dfs_history = []
    current_history_index = 0
    cur_df = None
    
    print('********************** start ***********************')
    
    run = True
    skip_update = False
    dataframe_update = False
    warn_message = []
    pickle_file_path = ''
    if 'auto_load' in settings:
        pickle_file_path = settings['auto_load']
        if os.path.exists(pickle_file_path):
            print(f'Loading dataframe from {os.path.abspath(pickle_file_path)}')
            cur_df = pd.read_pickle(pickle_file_path)
            view = update_view(cur_df,df_filters,view,categories)
            print(f'Loaded dataframe from {os.path.abspath(pickle_file_path)}')
            dfs_history.append(cur_df.copy())
            print_help('basic',view)
        else:
            print_help('start',view)
    else:
        print_help('start',view)
    # while loop
    while run:
        _input = input('')
        if len(_input) > 0:
            try:
                for fil,i in zip(settings['filters'],range(len(settings['filters']))):
                    if 'key' in fil:
                        if _input == fil['key']:
                            df_filters = fil
                            break
                for vie,i in zip(settings['views'],range(len(settings['views']))):
                    if 'key' in vie:
                        if _input == vie['key']:
                            view["type"] = vie['type']
                            view["view"] = i
                            view["page"] = '1'
                            if 'group' in vie: 
                                view["group"] = vie['group']
                            else:
                                view["group"] = None
                            break
                if len(df_filters) and view["type"] != df_filters["type"]:
                    df_filters = {}
                if view["type"][:min(6,len(view["type"]))] == 'budget':
                    if view["year"] == 0:
                        view["year"] = datetime.now().year
                    if view["year"] in budgets:
                        cur_df = budgets[view["year"]]
                        if view["group"] == 'category':
                            cur_df = group_and_sum_rows(cur_df,1)
                            if view["type"] == 'budget status':
                                post_dfs = dfs_history[current_history_index].copy()
                                cur_df = budget_status(cur_df,post_dfs)
                    else:              
                        warn_message.append(f'Budget {view["year"]} not found')
                        cur_df = None
                        skip_update = True
                elif len(dfs_history) and view['type'] == 'posts':
                    cur_df = dfs_history[current_history_index].copy()
                if _input in 'ur':
                    if view["type"] == 'plots':
                        if _input.lower() == 'u':
                            current_history_index += 1
                        elif _input.lower() == 'r':
                             current_history_index -= 1
                        if current_history_index < 0:
                            warn_message.append('Redo not possible. Current stage is the latest')
                            current_history_index = 0
                        elif current_history_index >= len(dfs_history):
                            warn_message.append('Undo not possible. No more history saved')
                            current_history_index = len(dfs_history) - 1
                if _input == 'clear':
                    print('Are you sure to clear all data?')
                    clear_input = input('y/n: ')
                    if clear_input == 'y':
                        dfs_history = []
                        cur_df = None
                        current_history_index = 0
                        skip_update = True
                if _input in 'wl':
                    pkl_path = pickle_file_path
                    if _input == 'w':
                        if os.path.exists(pkl_path):
                            print(f'Do you want to overwrite {pkl_path}?')
                            overwrite_input = input('y/n: ')
                        else:
                            overwrite_input = 'y'
                        if not overwrite_input == 'y':
                            pkl_path = filedialog.asksaveasfilename(defaultextension=".pkl",filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],title="Save As")
                    if _input == 'l':
                        if len(pkl_path) == 0 or not os.path.exists(pkl_path):
                            pkl_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
                            if len(pkl_path) == 0 or not os.path.exists(pkl_path):
                                skip_update = True
                                warn_message.append('Cancelling load')
                        if len(pkl_path) > 0:
                            if os.path.exists(pkl_path):
                                if len(dfs_history) > 0:
                                    print(f'Do you want to load {os.path.abspath(pkl_path)}?')
                                    load_input = input('Unsaved data will be deleted. y/n: ')
                                else:
                                    load_input = 'y'
                                if load_input == 'y':
                                    cur_df = pd.read_pickle(pkl_path)
                                    print(f'Loading dataframe from {os.path.abspath(pkl_path)}')
                                    pickle_file_path = pkl_path
                                else:
                                    skip_update = True
                                    warn_message.append('Cancelling load')
                            else:
                                skip_update = True
                                warn_message.append('Cancelling load')
                    elif _input.lower() == 'w':
                        cur_df.to_pickle(pkl_path)
                        print(f'Writing dataframe to {os.path.abspath(pkl_path)}')
                        pickle_file_path = pkl_path
                        print(f'ok')
                        skip_update = True
                # select import
                if _input == 'i':
                    dataframe_update = True
                    cur_df = import_data(cur_df)
                    if cur_df is None:
                        skip_update = True
                        warn_message.append('Import canceled')
                if _input[0] in 'pcfshdb' or _input.split(' ')[0] == 'auto' :
                    if _input[0] == 'h':
                        print_help('basic',view)
                        skip_update = True
                    # Pages
                    if _input[0] == 'p':
                        if _input == 'pn' or _input == 'pp':
                            if _input == 'pn':
                                add = 1
                            else:
                                add = -1
                            view["page"] = str(int(view["page"]) + add)
                        elif _input[0] == 'p':
                            try:
                                view["page"] = str(int(_input[1:]))
                            except:
                                skip_dummy = 0
                        else: 
                            view["page"] = '1'
                    # Year
                    if _input[0] == 'b':
                        if _input == 'bn' or _input == 'bp':
                            if _input == 'bn':
                                add = 1
                            else:
                                add = -1
                            view["year"] = int(view["year"]) + add
                    # Safe dataframe as own, and only update after all this
                    # select filters
                    if _input.split(' ')[0] == 'f':
                        _split = _input.split(' ')
                        if len(_split) == 1:
                            print_help('filter',view)
                            print()
                            print('Current:')
                            print(df_filters)
                            skip_update = True
                        if len(_split) == 2: 
                            if _split[1][:min(3,len(_split[1]))] == 'res':
                                df_filters = settings['filters'][view['filter']].copy()
                        if len(_split) == 3: 
                            if _split[1] == 'del':
                                _column = find_column(_split[2])
                                if _column in df_filters:
                                    del df_filters[_column]
                            elif _split[1] in ['sor','sor2','sort','sort2']:
                                if _split[1][-1] != '2':
                                    sort_name = 'sort'
                                else:
                                    sort_name = 'sort2'
                                if _split[2] == 'asc' or _split[2] == 'des':
                                    if _split[1][-1] != '2':
                                        df_filters[sort_name]['direction'] = _split[2]
                                    else:
                                        df_filters[sort_name]['direction'] = _split[2]
                                else:
                                    _column = find_column(_split[2])
                                    if _column is not None:
                                       if df_filters[sort_name]['column'] != _column:
                                           df_filters[sort_name]['column'] = _column
                                       else:
                                           if df_filters[sort_name]['direction'] == 'des':
                                               df_filters[sort_name]['direction'] = 'asc'
                                           else:
                                               df_filters[sort_name]['direction'] = 'des'
                            else:
                                _column = find_column(_split[1])
                                df_filters = change_filter(df_filters,_column,_split[2])
                    # select detail
                    if _input.split(' ')[0] == 'd':
                        _split = _input.split(' ')
                        if len(_split) == 1:
                            print_help('detail',view)
                        else:
                            change_ids = update_view(cur_df,df_filters,view,None,_split[1])
                            print(len(change_ids))
                            if len(change_ids) == 1:
                                print()
                                print('***Imported/fixed data columns')
                                print(f'id: ' + str(cur_df.loc[change_ids,'id'].values[0]))
                                print(f'posted_date: ' + str(cur_df.loc[change_ids,'posted_date'].dt.strftime('%d %b %Y').values[0]))
                                print(f'text: ' + cur_df.loc[change_ids,'text'].values[0])
                                print(f'value: ' + str(cur_df.loc[change_ids,'value'].values[0]))
                                print(f'balance: ' + str(cur_df.loc[change_ids,'balance'].values[0]))
                                print(f'account: ' + cur_df.loc[change_ids,'account'].values[0])
                                print(f'info: ' + cur_df.loc[change_ids,'info'].values[0])
                                print(f'imported: ' + str(cur_df.loc[change_ids,'imported'].dt.strftime('%d %b %Y %H:%M:%S').values[0]))
                                print('***Category/changable data')
                                print(f'date: ' + str(cur_df.loc[change_ids,'date'].dt.strftime('%d %b %Y').values[0]))
                                cat_typ = cur_df.loc[change_ids,'category_type'].values[0]
                                cat = cur_df.loc[change_ids,'category'].values[0]
                                print(f'category_type: ' + str(cat_typ))
                                print(f'category: ' + cat)
                                if cat_typ != '':
                                    category_name = categories[cat_typ]['name']
                                    sub_category_name = categories[cat_typ][cat]
                                    print(f'Category display name: {category_name} - {sub_category_name}')
                                print(f'amount: ' + str(cur_df.loc[change_ids,'amount'].values[0]))
                                print(f'split_id: ' + str(cur_df.loc[change_ids,'split_id'].values[0]))
                                print(f'marked: ' + cur_df.loc[change_ids,'marked'].values[0])
                                print(f'notes: ' + cur_df.loc[change_ids,'notes'].values[0])
                                print(f'status: ' + cur_df.loc[change_ids,'status'].values[0])
                                print(f'changed: ' + str(cur_df.loc[change_ids,'changed'].dt.strftime('%d %b %Y %H:%M:%S').values[0]))
                            else:
                                print('Only 1 selection allowed')
                        skip_update = True

                    # select change
                    elif _input.split(' ')[0] == 'c' or _input.split(' ')[0] == 'cd':
                        _split = _input.split(' ')
                        if len(_split) == 1:
                            print_help('change',view)
                            skip_update = True
                        elif len(_split) >= 4 or _input.split(' ')[0] == 'cd':
                            _column = find_column(_split[2],False)
                            if _column is not None:
                                change_ids = update_view(cur_df,df_filters,view,None,_split[1])
                                if len(change_ids) > 0:
                                    if _input.split(' ')[0] == 'cd':
                                        _value = ''
                                    else:
                                        if _column == 'notes':
                                            _value = ' '.join(_split[3:])
                                        else:
                                            _value = convert_value_to_data_type(cur_df,view,_column,_split[3])
                                    if _value is not None:
                                        if _column == 'category':
                                            if _value != '':
                                                _value = autocomplete_category(categories,_value)
                                            else:
                                                _value = ['','']
                                            if _value is not None:
                                                dataframe_update = True
                                                cur_df = change_dataframe_rows(cur_df,view,_column,change_ids,_value[1])
                                                cur_df = change_dataframe_rows(cur_df,view,'category_type',change_ids,_value[0])
                                            else: skip_update = True
                                        elif _column == 'marked':
                                            if _value != '':
                                                _value = autocomplete_marked(marked,_value)[0]
                                            if _value is not None:
                                                dataframe_update = True
                                                cur_df = change_dataframe_rows(cur_df,view,_column,change_ids,_value)
                                            else: skip_update = True
                                        else:
                                            dataframe_update = True
                                            cur_df = change_dataframe_rows(cur_df,view,_column,change_ids,_value)
                                    else: skip_update = True
                            else: skip_update = True
                    # select budget
                    if _input.split(' ')[0] == 'auto':
                        _split = _input.split(' ')
                        if len(_split) > 1:
                            id_sel = _split[1]
                        else:
                            id_sel = 'all'
                        change_ids = update_view(cur_df,df_filters,view,None,id_sel)
                        cur_df = auto_change_dataframe(cur_df,df_filters, view,auto_categories,change_ids)
                        dataframe_update = True
                    # Split
                    if _input.split(' ')[0] == 's' or _input.split(' ')[0] == 'sd':
                        _split = _input.split(' ')
                        if len(_split) == 1:
                            print_help('split',view)
                            skip_update = True
                        elif len(_split) >= 3 or _input.split(' ')[0] == 'sd': 
                            if _input.split(' ')[0] == 'sd':
                                change_ids = update_view(cur_df,df_filters,view,None,_split[1])
                                if len(change_ids) > 0:
                                    cur_df = delete_split_dataframe_rows(cur_df, change_ids)
                                else: skip_update = True
                            else:
                                change_ids = update_view(cur_df,df_filters,view,None,_split[1])
                                if len(change_ids) > 0:
                                    new_value = float(_split[2])
                                    if new_value is not None and new_value != 0:
                                        if len(_split) > 3 and len(_split[3]) > 0: #Create new_category
                                            new_cat = autocomplete_category(categories,_split[3])
                                            if not new_cat == None:
                                                no_cat = False
                                            else:
                                                no_cat = True
                                            while no_cat:
                                                print('No category found')
                                                _cat_q = input('Search again? y/n, or c for cancel: ')
                                                #_cat_q = input()
                                                if _cat_q[0].lower() == 'y':
                                                    _cat_input = input('search string: ')
                                                    new_cat = autocomplete_category(categories,_cat_input)
                                                    if not new_cat == None:
                                                        no_cat = False
                                                elif _cat_q[0].lower() == 'c':
                                                    new_cat = None
                                                    no_cat = False
                                                    print('Skipping command')
                                                elif _cat_q[0].lower() == 'n':
                                                    new_cat = ''
                                                    no_cat = False
                                        else:
                                            new_cat = '' # Splits to same category
                                        if new_cat is not None:
                                            if len(_split) > 4: #Create new_mark 
                                                new_mark = autocomplete_marked(marked,_split[4])
                                            else:
                                                new_mark = ''
                                            if len(_split) > 5: #Create note 
                                                if _split[5][0] == 'y':
                                                    print('Enter new note')
                                                    new_note = input('')
                                            else:
                                                new_note = ''
                                            dataframe_update = True
                                            cur_df = split_dataframe_rows(cur_df, change_ids, new_value, new_cat, new_mark, new_note)
                                        else: skip_update = True
                                    else: skip_update = True
                                else: skip_update = True

                # Updates view
                elif _input[0] == 'q':
                    if _input == 'qw':
                        _q_input = 'w'
                    elif _input == 'q!':
                        _q_input = 'y'
                    else:
                        print('Are you sure you want to quit without saving?')
                        _q_input = input('y/n or w for write and quit: ')
                    if _q_input[0].lower() == 'y':
                        run = False
                        skip_update = True
                    elif _q_input[0].lower() == 'w':
                        pkl_path = pickle_file_path
                        overwrite_input = 'n'
                        if len(pkl_path) == 0 or not pkl_path[-3:].lower() == 'pkl':
                            pkl_path = filedialog.asksaveasfilename(defaultextension=".pkl",filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],title="Save As")
                            if len(pkl_path) > 0:
                                overwrite_input = 'y'
                        else:
                            overwrite_input = 'y'
                        if overwrite_input == 'y':
                            print(f'Writing dataframe to {os.path.abspath(pkl_path)}')
                            dfs_history[current_history_index].to_pickle(pkl_path)
                            print('ok')
                            run = False
                            skip_update = True
                else:
                    #Update view
                    a = 0
                if not skip_update:
                    view = update_view(cur_df,df_filters,view,categories)
                else:
                    skip_update = False
                if dataframe_update:
                    if cur_df is not None:  
                        if view["type"] == 'posts':
                            if current_history_index > 0:
                                del dfs_history[:current_history_index]
                                current_history_index = 0
                            dfs_history.insert(0,cur_df.copy())
                            if len(dfs_history) > 10:
                                del dfs_history[10:]
                    dataframe_update = False
                #update dfs
            except Exception as e:
                # Get the traceback information as a string
                traceback_str = traceback.format_exc()
                # Print or log the traceback string
                warn_message.append(f"Error: {traceback_str}")
                if view["type"][:min(6,len(view["type"]))] == 'budget':
                    if view["year"] == 0:
                        view["year"] = datetime.now().year
                    if view["year"] in budgets:
                        cur_df = budgets[view["year"]]
                        if view["group"] == 'category':
                            cur_df = group_and_sum_rows(cur_df,1)
                elif len(dfs_history) and view['type'] == 'plots':
                    cur_df = dfs_history[current_history_index].copy()
                if len(cur_df):
                    view = update_view(cur_df,df_filters,view,categories)
                warn_message.append(f"Change command and try again.")
                warn_message.append('')
            for w in warn_message: print(w)
            warn_message = []

 
        readline.add_history(_input)
    
    
    print('********************** end ************************')
