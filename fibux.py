import subprocess
import importlib
import sys
import re
import os
import csv
import glob
import warnings
import ast
import calendar
import readline
import traceback
import shlex
from collections import defaultdict
import math as m
from datetime import datetime,timedelta
#test
#test
#test
#test
#test

required_modules = ['numpy', 'pandas', 'matplotlib', 'prettytable','tkinter','plotly','art','pynput','dateutil']
def check_modules(modules):
    missing_modules = []
    for module in modules:
        if not is_module_installed(module):
            missing_modules.append(module)
    return missing_modules

def is_module_installed(module_name):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def install_modules(modules):
    """Install a list of modules using the preferred pip command."""
    for module in modules:
        if module not in ['tkinter']:
            install_module(module)
        else:
            print(f'{module} cannot be installed using pip.')
            print('Must be installed manually')
            print()

def install_module(module_name):
    """Install a module using the preferred pip command."""
    print()
    try:
        print(f"Installing '{module_name}'...")
        subprocess.check_call([sys.executable, "-m","pip","install", module_name])
        print(f"'{module_name}' installed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to install '{module_name}'. Please try to install it manually.")

def display_missing_modules(modules):
    print()
    print("Missing Modules:")
    for module in modules:
        print(module)
    print()

if __name__ == '__main__':
    missing_modules = check_modules(required_modules)
    if missing_modules:
        print("The following modules are missing:")
        display_missing_modules(missing_modules)
        should_install = input("Do you want to install the missing modules? (y/n): ").strip().lower()
        if should_install == 'y':
            install_modules(missing_modules)
        else:
            print("Skipping the installation of missing modules. The script may not work properly without them.")
    else:
        print("All required modules are installed.")

import pandas as pd
import tkinter as tk
from tkinter import filedialog
from prettytable import PrettyTable as pt
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from art import *
from pynput.keyboard import Key, Listener
from dateutil.relativedelta import relativedelta

def create_pages(df,v,sel=None):
    pages = m.ceil(len(df)/int(settings["view_rows"]))
    total_length = len(df)
    # Dataframe to show
    page = int(v["page"]) 
    if page < 1:
        page = 1
    if page > pages:
        page = pages
    v["page"] = str(page)
    v["pages"] = str(pages)
    if pages > 1:
        if page < pages:
            df = df.iloc[0:page * int(settings['view_rows']),:]
        if page > 1: 
            df = df.iloc[(page - 1) * int(settings['view_rows']):,:]
    if sel is not None:
        return df.iloc[int(sel)-1,:],v
    else:
        return df,v

def create_table(df_in,v,col,ali=[],header='',output=print):
    df = df_print_format(df_in)
    max_col = v['max_col_width']
    table = pt(col)
    table._max_width = {c: max_col for c in col}
    for c, a in zip(col, ali):
        table.align[c] = a
    for index,row in df[col].iterrows():
        if not v['show_multiple_lines']:
            for column in col:
                if isinstance(row[column],str):
                    if len(row[column]) > max_col:
                        row[column] = row[column][:max_col-3] + '...'
        table.add_row(row)
    if output == 'print':
        print('')
        tprint(header,font="aquaplan")
        print(table)
    elif output == 'table':
        return table

def convert_to_bool(i):
    if isinstance(i,str):
        if i.lower() == 'true':
            return True
        elif i.lower() == 'false':
            return False
        else:
            return i
    else:
        return i

def format_plot_value(value):
    if abs(value) >= 10000:
        return f"{value / 1000:.1f}k"
    else:
        return f"{int(value)}"

def get_date_range(input_string):
    # Parse the number of months from the input string
    if input_string == 'rcyfmon':
        # Get the current date
        today = datetime.today()

        # Calculate the end date (last moment of the previous month)
        end_date = datetime(today.year, today.month, 1) - timedelta(seconds=1)
        start_date = datetime(end_date.year, 1, 1, 0, 0)
    else:
        match = re.match(r'rl(\d+)fmon', input_string)
        if not match:
            raise ValueError("Input string format is incorrect. Expected format 'l12fmon'.")

        num_months = int(match.group(1))
        print(f"Number of months: {num_months}")  # Debugging statement

        # Get the current date
        today = datetime.today()

        # Calculate the end date (last moment of the previous month)
        end_date = datetime(today.year, today.month, 1) - timedelta(seconds=1)
        print(f"End date: {end_date}")  # Debugging statement

        # Calculate the start date (first moment of the month 'num_months' ago)
        start_year = end_date.year
        start_month = end_date.month - num_months + 1
        if start_month <= 0:
            start_month += 12
            start_year -= 1
        print(f"Start year: {start_year}, Start month: {start_month}")  # Debugging statement

        start_date = datetime(start_year, start_month, 1, 0, 0)
        print(f"Start date: {start_date}")  # Debugging statement
    return [start_date, end_date]

def import_settings(_type):
    if _type == 'categories':
        # create an empty dictionary to store DataFrames for each line
        auto_categories = []
        categories = {}
        category_types = {}
        # open the text file for reading
        category_input = ''
        if os.path.exists("settings." + _type):
            with open("settings." + _type) as f:
                current_category = None
                current_subcategory = None
                
                for line in f:
                    if line[0] == '*':
                        _category_input = line[1:].strip() 
                    elif line[0] != '#':
                        if _category_input == 'Category_types':
                            _split = line.split(';')
                            if len(_split) >= 2:
                                if _split[0] == 'income':
                                    category_types['income'] = _split[1].strip()
                                if _split[0] == 'ignore':
                                    category_types['ignore'] = _split[1].strip()
                        if _category_input == 'Categories':
                            _split = line.split(';')
                            if len(_split) >= 2:
                                if _split[0].isdigit():
                                    current_category = int(_split[0].strip())
                                    categories[current_category] = {'name': _split[1].strip()}
                                if len(_split) > 2:
                                    categories[current_category].update({'type': _split[2].strip()})
                                else:
                                    current_subcategory = _split[0].strip()
                                    categories[current_category][current_subcategory] = _split[1].strip()
                        elif _category_input == 'Auto category':
                            change_values = {}
                            search_criteria = {}
                            _split = line.split(';')
                            auto_categories.append({'change':{},'search':{}})
                            split_characters = ['>', '<', '=']
                            for _spi in _split:
                                column = find_column(re.split(f'[{"".join(split_characters)}]', _spi)[0])
                                split_character = next(char for char in _spi if char in split_characters)
                                value = _spi.split(split_character)[1].strip()
                                if column in ['category','marked','date','notes']:
                                    auto_categories[-1]['change'][column] = value
                                elif column in ['value']:
                                    auto_categories[-1]['search'][column] = split_character + value
                                elif column in ['text','posted_date','info']:
                                    auto_categories[-1]['search'][column] = value

            
        return category_types,categories,auto_categories
    elif _type == 'marked':
        # create an empty dictionary to store DataFrames for each line
        marked = {}
        
        # open the text file for reading
        if os.path.exists("settings." + _type):
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
    elif _type == 'dept':
        loans = []
        if os.path.exists("settings." + _type):
            with open("settings." + _type) as f:
                content = f.read()

            all_lines = content.strip().split('\n')
            parameters = {}
            for line in all_lines:
                used_part = line.split('#')[0]
                if len(used_part):
                    if 'name:' == used_part[:5].lower():
                        if len(parameters):
                            loans.append(parameters)
                            parameters = {}
                        parameters = {}
                        parameters['name'] = line.strip().split(';')[0]
                        parameters['filter'] = line.strip().split(';')[1:]
                    elif len(parameters):
                        key = re.split(':|=',line,1)[0].strip()
                        value = re.split(':|=',line,1)[1].strip()
                        parameters[key] = value
            if len(parameters):
                loans.append(parameters)
            return loans
    elif _type == 'import' and False:
        # Convert the string representation to a dictionary using ast.literal_eval
        settings = {}
        if os.path.exists("settings." + _type):
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
        if os.path.exists("settings." + _type):
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
    elif _type == 'plots':
        settings = {}
        plots = [0]*100
        # Read content from the file
        if os.path.exists("settings." + _type):
            with open("settings." + _type) as f:
                content = f.read()
            exec(content)
            plots_out = []
            for p in plots:
                if p != 0:
                    plots_out.append(p)
            settings['plots'] = plots_out
        return settings
    elif _type == 'export':
        settings = {}
        setting_output = ''
        if os.path.exists("settings." + _type):
            with open("settings." + _type) as f:
                for line in f:
                    if len(line) > 6 and line[:7].lower() == 'output:':
                        setting_output = line.split(':')[1].strip() 
                        if setting_output not in settings:
                            settings[setting_output] = []
                    elif line[0] != '#':
                        settings[setting_output].append(line.strip())
        return settings
    else:
        settings = {}
        if os.path.exists("settings." + _type):
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
    #new_data = new_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    new_data = new_data.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # Remove the 'ignore' column if it exists
    for i in range(len(headers)):
        if 'ignore_' + str(i) in new_data.columns:
            new_data = new_data.drop('ignore_'+str(i), axis=1) 

    # convert value and balance to float
    new_data["value"] = new_data["value"].str.replace(thousand_sep, "").str.replace(decimal_sep, ".").astype(float)
    if 'amount' in new_data:
        new_data["amount"] = new_data["amount"].str.replace(thousand_sep, "").str.replace(decimal_sep, ".").astype(float)
    if 'category_type' in new_data:
        new_data["category_type"] = pd.to_numeric(new_data["category_type"],errors='coerce')
    new_data["balance"] = new_data["balance"].str.replace(thousand_sep, "").str.replace(decimal_sep, ".").astype(float)
    #new_data["value"] = new_data["value"].str.replace(thousand_sep, "", regex=True).str.replace(decimal_sep, ".", regex=True).astype(float)
    #if 'amount' in new_data:
    #    new_data["amount"] = new_data["amount"].str.replace(thousand_sep, "", regex=True).str.replace(decimal_sep, ".", regex=True).astype(float)
    #if 'category_type' in new_data:
    #    new_data["category_type"] = pd.to_numeric(new_data["category_type"],errors='coerce')
    #new_data["balance"] = new_data["balance"].str.replace(thousand_sep, "", regex=True).str.replace(decimal_sep, ".", regex=True).astype(float)
    
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
        #cur_df = check_and_correct_df(cur_df)
    data[add_col[1:]] = data[add_col[1:]].fillna('')
    data = check_and_correct_df(data)
    return data

def import_change_by_id(df):
    import_setting = import_settings('import_change')
    headers = import_setting["headers"]
    csv_sep = import_setting["csv_sep"]
    decimal_sep = import_setting["decimal_sep"]
    thousand_sep = import_setting["thousand_sep"]
    date_format = import_setting["date_format"]
    import_folder = import_setting.get("import_folder", None)
    csv_encoding = import_setting["csv_encoding"]

    found_csv = False
    # find newest csv file in folder
    if import_folder:
        latest_file = get_newest_csv_file(import_folder)
        if latest_file is not None:
            print(f'Use following file?')
            print(latest_file)
            ans = input('y/n: ')
            if ans == 'y':
                found_csv = True
    if not found_csv:
        root = tk.Tk()
        root.withdraw()
        latest_file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not latest_file:
            return None

    mod_df = df.copy()
    headers = []
    with open(latest_file, 'r') as f:
        for line in f:
            split_line = line.split(csv_sep)
            if len(headers) == 0:
                headers = [string.strip() for string in split_line]
                id_col = headers.index('id')
            elif len(split_line) > id_col:
                id_num = int(split_line[id_col].strip())
                change_input = ['id='+str(id_num)]
                for spl,col in zip(split_line,headers):
                    if col in ['notes']:
                        col_found = find_column(col,read_only=False)
                        if col_found != None:
                            change_input.extend([col])
                            change_input.extend([spl.strip()])
                            mod_df = change_dataframe_handle(change_input,mod_df)
    return mod_df

def import_data(input_df=None):
    if input_df is not None:
        if len(input_df) == 0:
            input_df = None
    print('')
    print('')
    if os.path.exists("settings.import"):
        import_setting = import_settings('import')
    else:
        print('settings.import does not exists')
        return input_df
    headers = import_setting["headers"]
    for i in range(len(headers)):
        if len(headers[i].lower()) > 5:
            if headers[i][:6].lower() == 'ignore':
                headers[i] = 'ignore_' + str(i)
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
            print(f'Use following file?')
            print(latest_file)
            ans = input('y/n: ')
            if ans == 'y':
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
    #new_data = add_missing_columns(new_data)

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
                initial_id = dfs_history[current_history_index]['id'].max() + 1
                new_data['id'] = range(initial_id,initial_id + len(new_data))
                if 'split_id' not in headers:
                    new_data['split_id'] = -1
                if 'amount' not in headers:
                    new_data['amount'] = new_data['value']
                if 'date' not in headers:
                    new_data['date'] = new_data['posted_date']
                new_data['account'] = _account
                new_data = add_missing_columns(new_data)

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
                            if input_df is None:
                                new_data['id'] = range(len(new_data))
                            else:
                                initial_id = dfs_history[current_history_index]['id'].max() + 1
                                new_data['id'] = range(initial_id,initial_id + len(new_data))
                            if 'split_id' not in headers:
                                new_data['split_id'] = -1
                            if 'amount' not in headers:
                                new_data['amount'] = new_data['value']
                            if 'date' not in headers:
                                new_data['date'] = new_data['posted_date']
                            new_data['account'] = account_input
                            new_data = add_missing_columns(new_data)
                            if df is None:
                                df = new_data
                            else:
                                df = pd.concat([df, new_data], ignore_index=True)
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
                if 'split_id' not in headers:
                    df['split_id'] = -1
                if 'amount' not in headers:
                    df['amount'] = df['value']
                if 'date' not in headers:
                    df['date'] = df['posted_date']
                df['account'] = account_input
                new_data = add_missing_columns(new_data)
                tprint("New dataframe",font="aquaplan")
                print(new_data)              
                match_found = True
        else:
            warnings.warn('Imported data needs minimum 3 rows')
    if df is not None and match_found == True:
        return df
    else:
        return None
def df_print_format(df):
    #try:
    float_format = lambda x: f"{x:,.2f}"
    for c in ['amount','value','balance']:
        if c in df.columns:
            df[c] = df[c].apply(float_format)
    float_format = lambda x: f"{x:,.0f}"
    for c in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','tot']:
        if c in df.columns:
            df[c] = df[c].apply(float_format)
    for c in ['posted_date','date','imported','changed']:
        if c in df:
            df[c] = df[c].dt.strftime(settings['date_format_print'])
    #except:
    #    pass
    return df

def create_output(df,df_fil,v,output_method=0,silent=False):
    # Select correct view
    print_col = settings['views'][v["view"]]['columns']
    print_col_align = settings['views'][v["view"]]['columns']
    if v["type"] == 'budget status':
        if v['year'] == datetime.now().year:
            cur_mon = datetime.now().month
            if cur_mon < 12:
                rem_month_col = get_months(range(cur_mon+1,13))
                for rem_mon in rem_month_col:
                    if rem_mon in print_col:
                        print_col.remove(rem_mon)
            if cur_mon == 1 and 'tot' in print_col:
                print_col.remove('tot')
    align_col = settings['views'][v["view"]]['align']
    # Set filters based on type
    filters = 0
    # Get filtered dataframe
    df_filtered = filter_dataframe(df, df_fil)
    if df_fil != None:
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
    else:
        df_sorted = df_filtered.copy()

    if 'budget' in v['type']:
        if v['seperate_marked_in_budget']:
            mar_list = autocomplete_marked()
            #mar_dic = {index: value for index, value in enumerate(mar_list)}
            mar_dic = {key: {} for key in mar_list}
            new_id = 0
            for _m,_item in settings['marked'].items():
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
            for _m,_item in settings['marked'].items():
                post_df = dfs_history[current_history_index].copy()
                max_date = datetime.strptime(str(v['year']-1)+'-12-31', '%Y-%m-%d')
                #min_date = datetime.strptime(str(v['year']-1)+'-12-1', '%Y-%m-%d')
                marked_filter = {"key": "", "type": "posts","marked":{"equal":[_m],'empty':False},"date":{"max":max_date}}
                post_filtered = filter_dataframe(post_df, marked_filter)
                past_years_used = post_filtered['amount'].sum()
                mar_dic[_m]['jan'] -= past_years_used
                new_id -= 1
                mar_dic[_m]['id'] = new_id
            #new_rows = pd.DataFrame.from_dict({(i, 'name'): j for i, j in mar_dic.items()}, orient='index')
            # Convert the dictionary to a DataFrame
            new_rows = pd.DataFrame.from_dict(mar_dic, orient='index')
            new_rows['category'] = 'Marked'
            new_rows['year'] = v['year']

            # Reset the index and rename the 'index' column to 'name'
            new_rows = new_rows.reset_index().rename(columns={'index': 'name'})
            
            # Create a list of datetimes with a monthly frequency
            #new_rows = pd.DataFrame({'name': name_values})
            #df_sorted = pd.concat([new_rows, df_sorted], ignore_index=True)
            #df_sorted = pd.concat([df_sorted,new_rows], ignore_index=True)
            df_sorted = pd.concat([new_rows,df_sorted], ignore_index=True)
        _month_col = get_months()

    if v["type"] == 'budget status':
        if v['year'] == datetime.now().year:
            cur_mon = datetime.now().month
            _month_col = get_months(range(1,cur_mon+1))
            all_months = get_months()
            if len(_month_col) != len(all_months):
                for column_to_remove in all_months[len(_month_col):]:
                    df_sorted = df_sorted.drop(columns=[column_to_remove])
        post_dfs = dfs_history[current_history_index].copy()
        if output_method == 'df':
            df_sorted = budget_status(df_sorted,post_dfs,v,output_method)
        elif output_method != 0:
            if len(output_method) != 0:
                output_method_num = output_method[0]
                if len(output_method) > 1:
                    output_method_mon = (output_method[1])
                else:
                    output_method_mon = 'all'
            df_sorted = budget_status(df_sorted,post_dfs,v,[output_method_num,output_method_mon])
        else:
            df_sorted = budget_status(df_sorted,post_dfs,v)
    if 'budget' in v['type']:
        df_sorted['tot'] = df_sorted[_month_col].sum(axis=1)
    if 'budget' in v['type']:
        numeric_columns = df_sorted.select_dtypes(include='number')
        #columns_to_set_empty = [col for col in df_sorted.columns if col not in numeric_columns]
        df_total = pd.DataFrame(columns=df_sorted.columns)
        #df_total = df_total.append(pd.Series(dtype='object'), ignore_index=True)
        
        # Add a row with empty values to all columns
        #df_total.loc[0] = [''] * len(df_total.columns)
        
        #for c in columns_to_set_empty:
        #    df_total[c][0] = ''
        df_total.loc[0] = numeric_columns.sum()
        df_total.fillna('', inplace=True)
    if output_method != 'df' and output_method != 'table':
        if output_method != 'all':
             df_sorted,v = create_pages(df_sorted,v)
    if len(df_sorted) == 0 and not silent:
        print('Filter does not return any posts')
        print(df_fil)
        #return df_sorted
    
    if output_method == 0 or output_method == 'df' or output_method == 'table':
        #for date_column in ['posted_date','date','imported','changed']:
        #    if date_column in df_sorted and date_column in print_col:
        #        df_sorted[date_column] = df_sorted[date_column].dt.strftime('%d %b %Y')
        # Changing category column to name.
        
        if 'category' in settings['views'][v['view']]['columns']:
            for index, row in df_sorted.iterrows():
                category_type = row['category_type']
                category = row['category']
                if len(str(category_type)):
                    if isinstance(category_type,int) or isinstance(category_type,str):
                        if isinstance(category_type,str):
                            category_type = int(category_type.split(',')[0])
                if len(category) > 0:
                    if isinstance(row["category_type"],str) and len(row["category_type"].split(',')) > 1:
                        unique_types = list(set(row["category_type"].split(',')))
                        if len(unique_types) == 1:
                            category_name = settings['categories'][int(unique_types[0])]['name']
                        else:
                            category_name = 'Multiple'
                        # if they are the same as below otherwise Multiple
                    elif category == 'Marked':
                        category_name = 'Marked'
                    else:
                        category_name = 'Type error'
                        if 'categories' in settings:
                            if category_type in settings['categories']:
                                if 'name' in settings['categories'][category_type]:
                                    category_name = settings['categories'][category_type]['name']
                    if category_name == 'Multiple' or category_name == 'Marked':
                        sub_category_name = ''
                    else:
                        if len(category.split(',')) > 1:
                            sub_category_name = 'Multiple'
                        else:
                            sub_category_name = 'Category error'
                            if 'categories' in settings:
                                if category_type in settings['categories']:
                                    if category in settings['categories'][category_type]:
                                        sub_category_name = settings['categories'][category_type][category]
                    if len(sub_category_name):
                        _text = f"{category_name} - {sub_category_name}"
                    else:
                        _text = f"{category_name}"
                    #_text = category_name + ' - ' + sub_category_name
                    df_sorted.at[index, 'category'] = _text.strip()
                    # Select only numeric columns for summing
        #if len(df_sorted) > 0 and (output_method == 0 or output_method == 'table'):
            #df_sorted = df_print_format(df_sorted)
            #if 'budget' in v['type']:
            #    df_total = df_print_format(df_total)
            #  
            #float_format = lambda x: f"{x:,.2f}"
            #for c in ['amount','value','balance']:
            #    if c in df_sorted.columns:
            #        df_sorted[c] = df_sorted[c].apply(float_format)
            #float_format = lambda x: f"{x:,.0f}"
            #for c in ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec','tot']:
            #    if c in df_sorted.columns:
            #        df_sorted[c] = df_sorted[c].apply(float_format)
            #    if 'budget' in v['type']:
            #        if c in df_total.columns:
            #            df_total[c] = df_total[c].apply(float_format)
        if output_method == 'df':
            return df_sorted
        if output_method != 'table':
            df_sorted.loc[:,'select'] = range(1,len(df_sorted) + 1)
            print_col = list(['select'] + print_col)
            align_col = list(['c'] + align_col)
            if 'budget' in v['type']:
                df_total.loc[0,'select'] = ''
        if 'budget' in v['type']:
            df_sorted = pd.concat([df_sorted, df_total], ignore_index=True)
        if output_method == 0:
            print_header = settings['views'][v['view']]['name'].format(year=v['year'])
            create_table(df_sorted,v,header=print_header,col=print_col,ali=align_col,output='print')
            print("Page " + v['page'] + " of " + v['pages'])
            return v
        elif output_method == 'table':
            return create_table(df_sorted,v,col=print_col,ali=align_col,output='table')
    else:
        if v['type'] == 'posts':
            # Get and return row number
            if output_method == 'all':
                return df_sorted['id']
            else:
                _list = convert_string_to_list(output_method)
                n = len(df_sorted)  # number of rows in the DataFrame
                sel_list = [-1 + i for i in _list]  # convert to zero-based indices
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
    if df_filter != None:
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
    df_unique = filtered_df.drop_duplicates(subset='id', keep='last')
    if not get_ids: 
        return df_unique
    else:
        #not used
        return df_unique['id']

def find_column(input_string,read_only=True):
    # Define a dictionary of predefined words with their corresponding 3-letter keys
    if read_only:
        words = {
            'id': 'id',
            'tex': 'text',
            'inf': 'info',
            'not': 'notes',
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
            'inf': 'info',
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

def change_dataframe_rows(df, column_to_change, dataframe_ids, new_value):
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
    if new_value == 'first' and column_to_change == 'date':
        df.loc[df['id'].isin(ids_to_change),column_to_change] = df.loc[df['id'].isin(ids_to_change),'posted_date'] + pd.offsets.MonthBegin(1) 
    else:
        # Update the original DataFrame with the modified rows
        df.loc[df['id'].isin(ids_to_change),column_to_change] = new_value
        df.loc[df['id'].isin(ids_to_change),'changed'] = pd.to_datetime(datetime.now())

    return df 

def auto_change_dataframe(df, df_filt,cur_view, input_ids):
    mod_df = df.copy()
    for auto_value in settings['auto_categories']: 
        mod_filt = df_filt.copy()
        for _c,_v in auto_value['search'].items():
            mod_filt = change_filter(mod_filt,_c,_v)
        filter_ids = create_output(df, mod_filt, cur_view,'all',silent=True)
        combined_ids = pd.concat([input_ids, filter_ids],ignore_index=True)
        change_ids = combined_ids[combined_ids.duplicated()]
        for _c,_v in auto_value['change'].items():
           _value = convert_value_to_data_type(df,_c,_v)
           if _value is not None or _v == 'first':
               if _c == 'category':
                   if _v != '':
                       _v = autocomplete_category(_v)
                   else:
                       _v = ['','']
                   if _v is not None:
                       mod_df = change_dataframe_rows(mod_df,_c,change_ids,_v[1])
                       mod_df = change_dataframe_rows(mod_df,'category_type',change_ids,_v[0])
               elif _c == 'marked':
                   if _v != '':
                       _m = autocomplete_marked(_v)
                       if _m is not None:
                           _v = _m[0]
                   if _v is not None:
                       mod_df = change_dataframe_rows(mod_df,_c,change_ids,_v)
               else:
                   mod_df = change_dataframe_rows(mod_df,_c,change_ids,_v)
        
        #Change all change_ids in mod_df
    return mod_df

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
    max_id = dfs_history[current_history_index]['id'].max()

    # Create new rows with the updated values and unique IDs
    #new_rows = df.copy()
    new_rows = df.loc[df['id'].isin(ids_to_split)].copy()
    new_ids = range(max_id + 1, max_id + 1 + len(new_rows))
    new_rows['split_id'] = new_rows.apply(lambda row: row['split_id'] if row['split_id'] >= 0 else row['id'], axis=1)
    new_rows['id'] = new_ids
    if new_mark is not None:
        if new_mark != '-':
            if list(new_mark):
                new_rows['marked'] = new_mark[0]
            else:
                new_rows['marked'] = new_mark
    else:
        new_rows['marked'] = ''
    if new_category != '-':
        if new_category == '':
            new_rows['category_type'] = new_category
            new_rows['category'] = new_category
        else:
            new_rows['category_type'] = new_category[0]
            new_rows['category'] = new_category[1]
    if new_note is None:
        #new_rows['notes'] = df.loc[df['id'].isin(ids_to_split), 'notes'].str.cat(sep=' ')
        new_rows['notes'] = df.loc[df['id'].isin(ids_to_split), 'notes']
    else:
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

    global dataframe_update
    global skip_update
    global warn_message
    if len(df[((df['id'].isin(ids_to_delete)) & (df['split_id'] >= 0))]):
        for index, row in df[((df['id'].isin(ids_to_delete)) & (df['split_id'] >= 0))].iterrows():
            df_id = row['split_id']
            value_to_add = row['amount']
            
            # Find rows where account_id matches and split is 'n'
            row_to_update = df[df['id'] == df_id]
            
            # Update the 'value' column in the selected rows
            df.loc[row_to_update.index, 'amount'] += value_to_add
        
        df = df[~((df['id'].isin(ids_to_delete)) & (df['split_id'] >= 0))]
        return df
    else:
        skip_update = True
        dataframe_update = False
        warn_message.append('Did not find any split post to delete. And original post cannot be deleted')

def plot_dataframe(df, plot, settings):
    #plot_df = {'type':'sankey','df_filter':df_filt,'plot_options':{'monthly':True}}
    income_type = settings['category_types']['income']
    ignore_type = settings['category_types']['ignore']
    if 'df_filter' in plot and len(plot['df_filter']):
        plot_filter = plot['df_filter']
    else:
        plot_filter = {}
    if 'filter_commands' in plot:
        for fc in plot['filter_commands']:
            split_input = shlex.split(fc)
            plot_filter = set_filter(split_input,plot_filter)
    df_filtered = filter_dataframe(df, plot_filter)
    if 'options' in plot:
        options = plot['options']
    else:
        options = {}
    mode = plot['type']
    print(f'Plotting {mode}...')
    if mode in ['sankey','overview']:
        date1 = df_filtered['date'].max()
        date2 = df_filtered['date'].min()
        min_month_year = date2.strftime('%d-%b-%Y')
        max_month_year = date1.strftime('%d-%b-%Y')
        date_range_str = f"{min_month_year} to {max_month_year}"
        total_months = (date1.year - date2.year) * 12 + (date1.month - date2.month) + 1 
    if mode == 'overview':
        overview_stacks = {}
        #overview_stacks = {income_type:[0]*total_months}
        
        for i in range(total_months):
            start_date = datetime(date2.year,date2.month,1,0,0) + relativedelta(months=i)
            end_date = start_date + relativedelta(months=1) - timedelta(seconds=1)
            df_filtered_month = df_filtered[(df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)]
            for key, value in settings['categories'].items():
                if value['type'] != ignore_type and len(df_filtered_month[df_filtered_month['category_type'] == key]) > 0:
                    if value['type'] not in overview_stacks:
                        overview_stacks[value['type']] = [0]*total_months
                    _sum = df_filtered_month[df_filtered_month['category_type'] == key]['amount'].sum()
                    overview_stacks[value['type']][i] += _sum
        

        # Create a list of months for x-axis labels
        months = [start_date.strftime("%b-%Y") for start_date in pd.date_range(date2, periods=total_months, freq='ME')]
        # Create a list of stacked expense values for each month
        #stacked_expense_values = [overview_stacks[key] for key in overview_stacks if key != income_type]

        # Create a list of income and expense values for each month
        #income_values = overview_stacks[income_type]
        #expense_values = [sum(overview_stacks[key]) for key in overview_stacks if key != income_type]
        # Create a stacked plot
        fig, ax = plt.subplots()
        ax.grid(True,zorder=5)
        
        # Plot Indkomst on the positive side
        if income_type in overview_stacks:
            ax.bar(months, overview_stacks[income_type], label=income_type, color='forestgreen',zorder=5)
        
        # Initialize the bottom for stacking negative values
        negative_stack = np.zeros(len(months))
        
        # Plot the rest of the categories on the negative side
        colors = ['darkblue','darkred','darkorange','darkviolet','saddlebrown','darkcyan']
        count = 0
        for category, values in overview_stacks.items():
            if category != income_type:
                if len(colors) > count:
                    ax.bar(months, values, bottom=negative_stack, label=category, color=colors[count],zorder=5)
                else:
                    ax.bar(months, values, bottom=negative_stack, label=category,zorder=5)
                negative_stack += np.array(values)
                count += 1
        
        # Labels and title
        #ax.set_xlabel('Month')
        ax.set_ylabel('Amount')
        ax.set_xlabel(f"Post dates: {date_range_str}")
        if 'name' in plot:
            ax.set_title(plot['name'])
            #ax.sup_title(date_range_str)
        else:
            ax.set_title(f"Overview of monthly transactions")
        ax.legend()
        plt.xticks(rotation=45)
        
        plt.savefig('overview_plot.png')
        plt.show()
        print('overview_plot.png saved')
    elif mode == 'sankey':
        if 'monthly' in options and options['monthly']:
            # Calculate difference in months
            divider = total_months
        else:
            divider = 1.0
        # Calculate the total income and total expenses
        total_income = 0
        total_expenses = 0
        income_found = False
        expense_found = False
        expense_types = []
        for key, value in settings['categories'].items():
            if value['type'] != ignore_type:
                if value['type'] == income_type:
                    if len(df_filtered[df_filtered['category_type'] == key]['amount']):
                        _sum = df_filtered[df_filtered['category_type'] == key]['amount'].sum()
                        total_income += _sum
                        income_found = True
                else:
                    if value['type'] not in expense_types:
                        expense_types.append(value['type'])
                    if len(df_filtered[df_filtered['category_type'] == key]['amount']):
                        _sum = df_filtered[df_filtered['category_type'] == key]['amount'].sum()
                        total_expenses -= _sum
                        expense_found = True
        # Prepare the data for the Sankey diagram
        labels = []
        
        category_dict = {}
        exp_type_dict = {}
        cat_type_dict = {}
        cat_num_dict = {}
        
        # Add 'Total Income' and 'Total Expenses' to labels
        if income_found:
            income_index = len(labels)
            labels.append(f'Income\n{format_plot_value(total_income/divider)}')
        if expense_found:
            expense_index = len(labels)
            labels.append(f'Expenses\n{format_plot_value(total_expenses/divider)}')
        if income_found and expense_found:
            if total_income > total_expenses:
                buffer_index = len(labels)
                labels.append(f'Buffer\n{format_plot_value((total_income - total_expenses)/divider)}')
            elif total_expenses > total_income:
                buffer_index = len(labels)
                labels.append(f'Overconsumption\n{format_plot_value((total_expenses - total_income)/divider)}')

        
        # Build the dictionaries for expense types, category types and categories
        for exp_t in expense_types:
            exp_sum = 0
            for key, value in settings['categories'].items():
                if value['type'] == exp_t:
                    if len(df_filtered[df_filtered['category_type'] == key]) > 0:
                        exp_type_name = exp_t
                        exp_sum -= df_filtered[df_filtered['category_type'] == key]['amount'].sum()
            if abs(exp_sum) > 0:
                exp_type_dict[exp_t] = len(labels)
                labels.append(f"{exp_type_name}\n{format_plot_value(exp_sum/divider)}")
        for key, value in settings['categories'].items():
            if value['type'] != ignore_type:
                if len(df_filtered[df_filtered['category_type'] == key]) > 0:
                    cat_type_name = value['name']
                    cat_type_dict[key] = len(labels)
                    cat_type_sum = -1.0*df_filtered[df_filtered['category_type'] == key]['amount'].sum()
                    labels.append(f"{cat_type_name}\n{format_plot_value(cat_type_sum/divider)}")
        for key, value in settings['categories'].items():
            if value['type'] != ignore_type:
                if value['type'] == income_type:
                    mult = 1.0
                else:
                    mult = -1.0
                if len(df_filtered[df_filtered['category_type'] == key]) > 0:
                    for cat_key, cat_name in value.items():
                        if cat_key not in ['name', 'type']:
                            if len(df_filtered[(df_filtered['category_type'] == key) & (df_filtered['category'] == cat_key)]) > 0:
                                category_key = f"{key}_{cat_key}"
                                category_sum = mult*df_filtered[(df_filtered['category_type'] == key) & (df_filtered['category'] == cat_key)]['amount'].sum()
                                category_dict[category_key] = len(labels)
                                labels.append(f"{cat_name}\n{format_plot_value(category_sum/divider)}")
        sources = [0]*len(labels)
        targets = [0]*len(labels)
        values = [0]*len(labels)
        # First flows: Income categories to 'Total Income'
        if income_found:
            for key, value in settings['categories'].items():
                if value['type'] == income_type:
                    for _key, _value in settings['categories'][key].items():
                        if _key not in ['name','type']:
                            category_key = f"{key}_{_key}"
                            if category_key in category_dict:
                                income_sum = df_filtered[(df_filtered['category_type'] == key) & (df_filtered['category'] == _key)]['amount'].sum()
                                #sources.append(category_dict[category_key])
                                sources[category_dict[category_key]] = category_dict[category_key]  # Index of 'Total Income'
                                targets[category_dict[category_key]] = 0  # Index of 'Total Income'
                                values[category_dict[category_key]] = income_sum/divider
        # Next flow: 'Total Income' to 'Total Expenses'
        if income_found and expense_found:
            sources[income_index] = income_index  # Index of 'Total Income'
            targets[income_index] = expense_index  # Index of 'Total Expenses'
            values[income_index] = total_income/divider
            if total_income > total_expenses:
                values[0] = total_expenses/divider
                sources[buffer_index] = income_index
                targets[buffer_index] = buffer_index
                values[buffer_index] = (total_income - total_expenses)/divider
            elif total_expenses > total_income:
                values[0] = total_income/divider
                sources[buffer_index] = buffer_index
                targets[buffer_index] = expense_index
                values[buffer_index] = (total_expenses - total_income)/divider
        
        # Final flows: 'Total Expenses' to category types and categories
        if expense_found:
            for exp_t in expense_types:
                exp_sum = 0
                for key, value in settings['categories'].items():
                    if value['type'] == exp_t:
                        if len(df_filtered[df_filtered['category_type'] == key]) > 0:
                            exp_type_name = exp_t
                            exp_sum += df_filtered[df_filtered['category_type'] == key]['amount'].sum()
                if abs(exp_sum) > 0:
                    labels.append(f"{exp_type_name}\n{format_plot_value(exp_sum)}")
                    sources[exp_type_dict[exp_t]] = expense_index
                    targets[exp_type_dict[exp_t]] = exp_type_dict[exp_t] 
                    values[exp_type_dict[exp_t]] = -1*exp_sum/divider
            for key, value in settings['categories'].items():
                if value['type'] != income_type:
                    if key in cat_type_dict: 
                        expense_sum = df_filtered[(df_filtered['category_type'] == key)]['amount'].sum()
                        sources[cat_type_dict[key]] = exp_type_dict[value['type']] 
                        targets[cat_type_dict[key]] = cat_type_dict[key]
                        values[cat_type_dict[key]] = -1*expense_sum/divider
                        for _key, _value in settings['categories'][key].items():
                            if _key not in ['name','type']:
                                category_key = f"{key}_{_key}"
                                if category_key in category_dict:
                                    expense_sum = df_filtered[(df_filtered['category_type'] == key) & (df_filtered['category'] == _key) ]['amount'].sum()
                                    sources[category_dict[category_key]] = cat_type_dict[key]
                                    targets[category_dict[category_key]] = category_dict[category_key]
                                    values[category_dict[category_key]] = -1*expense_sum/divider
        # Create the Sankey diagram
        fig = go.Figure(go.Sankey(
            arrangement = "snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            )
        ))

        # Update layout
        if 'name' in plot:
            _title = plot['name']
        else:
            if 'monthly' in options and options['monthly']:
                _title = f"Sankey Diagram for {str(divider)} months" 
            else:
                _title=f"Sankey Diagram"
        title=f"{_title}<br>{date_range_str}"
        fig.update_layout(
            title_text=title,
            title_font_size=20,
            font_size=10
        )
        
        #fig.show()
        fig.write_html('sankey_plot.html', auto_open=True)

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

def convert_value_to_data_type(dfs, column, value):
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

def autocomplete_category( string_bit, output='default'):
    if string_bit != '':
        output2 = output
        if output == 'comp_list' or output == 'sep_list':
            output2 = 'list'
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
                if category in settings['categories']:
                    if subcategory in settings['categories'][category]:
                        if output2 == 'list':
                            count += 1
                            result.append(str(category) + subcategory)
                        elif output2 == 'default':
                            return [category, subcategory]
                    elif subcategory == '':
                        if output2 == 'list':
                            for sub, item in settings['categories'][category].items():
                                if sub != 'name' and sub != 'type':
                                    count += 1
                                    result.append(str(category) + sub)
                elif category == 0:
                    for cat, item in settings['categories'].items():
                        _type = ''
                        for sub, item in settings['categories'][cat].items():
                            if sub == 'type':
                                _type = item
                            if sub != 'name' and sub != 'type' and _type != settings['category_types']['ignore']:
                                count += 1
                                result.append(str(cat) + sub)
        # Check for matches in category names and subcategory names
        cats = []
        if not string_bit.isdigit():
            for category, subcategories in settings['categories'].items():
                for subcategory, name in subcategories.items():
                    if string_bit.lower() in name.lower():
                        if subcategory != 'name' and subcategory != 'type':
                            count += 1
                            if output2 == 'list':
                                result.append(str(category) + subcategory)
                            elif output2 == 'default':
                                result = [category, subcategory]
                            cats.append(result)
                        else:
                            if output2 == 'list':
                                for subcategory, name in subcategories.items():
                                    if subcategory != 'name' and subcategory != 'type':
                                        count += 1
                                        result.append(str(category) + subcategory)
                                        cats.append(result[-1])
        if count == 0:
            print('No category match found for ' + string_bit)
        elif count == 1:
            return result
        elif output == 'comp_list':
            return result
        elif output == 'sep_list':
            result1 = []
            result2 = []
            
            # Iterate through the original list and split each string
            for item in result:
                result1.append(item[:-1])
                result2.append(item[-1])
            return result1,result2
        else:
            print(f'{count} matches found for {string_bit}')
            try:
                for cat in cats:
                    print(f'  {str(cat[0])}{cat[1]}: {settings["categories"][cat[0]]["name"]} - {settings["categories"][cat[0]][cat[1]]}')
            except:
                pass
        for category, subcategories in settings['categories'].items():
            for subcategory, name in subcategories.items():
                if string_bit.lower() in name.lower():
                    A = 1
        
        # Return None if no match is found
    return None

def autocomplete_marked(string_bit='',output='default'):
    if 'marked' in settings:
        mark_list = []
        if string_bit != '':
            for mark, name in settings['marked'].items():
                if string_bit.lower() == mark[:min(len(string_bit),len(mark))].lower():
                    mark_list.append(mark)
            if len(mark_list) == 0:
                print('No marked options found for ' + string_bit)
            elif output == 'comp_list' or len(mark_list) == 1:
                return mark_list
            else:
                print(f'{len(mark_list)} matches found for {string_bit}')
        else:
            for mark, name in settings['marked'].items():
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
                    if input_value[1] == 'r':
                        _value = get_date_range(input_value[1:])
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
            if isinstance(_value,list):
                input_filters.setdefault(input_column, {})['min'] = _value[0]
                input_filters.setdefault(input_column, {})['max'] = _value[1]
            else:
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
                        for key in settings['categories'][int(ci)]:
                            if key != 'name':
                                cat_values.append(ci+key)
                    else:
                        cat_value_list = autocomplete_category(ci,output='comp_list')
                        if cat_value_list is not None:
                            for cvl in cat_value_list:
                                if cvl not in cat_values:
                                    cat_values.append(cvl)
                if len(cat_values) > 0:
                    input_filters.setdefault(input_column, {})['equal'] = cat_values
                    input_filters.setdefault(input_column, {})['not_empty'] = True
            elif input_column == 'marked':
                mar_values = []
                mar_ids = _value.split(',')
                for mi in mar_ids:
                    mar_value_list = autocomplete_marked(mi,output='comp_list')
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

def convert_delta_filter_time(settings):
    if 'filters' in settings:
        for i in range(len(settings['filters'])):
            for column_name, column_filter in settings['filters'][i].items():
                if column_name in ['posted_date','date','imported','changed']:
                    if 'date' in column_name or 'imported' in column_name:
                        for bound, time_val in column_filter.items():
                            if '-' == time_val[0]:
                                new_val = parse_duration_string(time_val)
                                settings['filters'][i][column_name][bound] = new_val
    return settings

def get_months(_i='all'):
    _all = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    if _i == 'all':
        return _all
    elif isinstance(_i,int):
        return _all[_i-1]
    elif isinstance(_i,str):
        if _i.isdigit():
            return _all[int(_i)-1]
        elif _i in _all:
            return _all.index(_i) + 1
    elif isinstance(_i,list) or isinstance(_i,range):
        output = []
        for __i in _i:
            output.append(_all[__i-1])
        return output

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
                all_auto_cat_list = []
                cat_sort = ''
                if len(row) and row[0][0] not in '#':
                    for cat in row[1].split(';'):
                        if cat[0] != '-':
                            cat_value_list = autocomplete_category(cat,output='comp_list')
                            if not cat_value_list == None:
                                for cat2 in cat_value_list:
                                    _t = int(cat2[:len(cat2)-1])
                                    _c = cat[-1]
                                    all_type += cat2[:len(cat2)-1]+','
                                    all_type_list.append(cat2[:len(cat2)-1])
                                    all_cat_list.append(cat2[-1])
                                    all_category += cat2[-1]+','
                                    cat_sort += "{:02d}".format(_t) + _c + ','
                        else:
                            all_auto_cat_list.append(cat)
                            cat_sort += row[1]
                    _id += 1
                    if len(all_auto_cat_list) == 0:
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
                    else:
                        row_data = {
                            'year': year,
                            'name': row[0],
                            'categories': row[1],
                            'category_type': '',
                            'category': '',
                            'category_sort': '!' + cat_sort,
                            'tot': 0,
                            'id': _id,
                            # Add other columns as needed
                        }
                    row_data.update({f'{month}': 0.0 for month in month_columns})
                    if len(row) == 14:
                        row_data.update({f'{month}': float(value) for month, value in zip(month_columns, row[2:])})
                    elif len(row) == 4:
                        for mon, ind in zip(row[2].split(';'), range(1000)):
                            values = row[3].split(';')
                            if ind >= len(values):
                                val = values[-1]
                            else:
                                val = values[ind]
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
            # Iterate through each key in the dictionary
            other_cat_type_list = []
            other_cat_sub_list = []
            for other_values in dfs:
                other_cat_type_list.extend(other_values.get('category_type', '').split(','))
                other_cat_sub_list.extend(other_values.get('category', '').split(','))

            zero_id = -1
            for values in dfs:
                # Check if the 'category' key exists and its first character is '-'
                if 'category' in values and len(values['categories']) and values['categories'][0] == '-':
                    cat_type_list,cat_sub_list = autocomplete_category(values['categories'][1:],output='sep_list')
                    if values['categories'][1:] == '0':
                        zero_id = values['id']
                    else:
                        # Check if the combination exists in other parts
                        for cat_type,cat_sub in zip(cat_type_list,cat_sub_list):
                            found = False
                            for other_cat_type,other_cat_sub in zip(other_cat_type_list,other_cat_sub_list):
                                if cat_type == other_cat_type and cat_sub == other_cat_sub:
                                    found = True
                                    break
                            if not found:
                                if values['category_type']:
                                    values['category_type'] += ',' + cat_type
                                else:
                                    values['category_type'] = cat_type
                                if values['category']:
                                    values['category'] += ',' + cat_sub
                                else:
                                    values['category'] = cat_sub
                                other_cat_type_list.append(cat_type) 
                                other_cat_sub_list.append(cat_sub) 
            if zero_id >= 0:
                for values in dfs:
                    if values['id'] == zero_id:
                        cat_type_list,cat_sub_list = autocomplete_category('0',output='sep_list')
                        for cat_type,cat_sub in zip(cat_type_list,cat_sub_list):
                            found = False
                            for other_cat_type,other_cat_sub in zip(other_cat_type_list,other_cat_sub_list):
                                if cat_type == other_cat_type and cat_sub == other_cat_sub:
                                    found = True
                                    break
                            if not found:
                                if values['category_type']:
                                    values['category_type'] += ',' + cat_type
                                else:
                                    values['category_type'] = cat_type
                                if values['category']:
                                    values['category'] += ',' + cat_sub
                                else:
                                    values['category'] = cat_sub

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
        'id': 'first'
    })
    grouped_df.rename(columns={'year': 'year', 'categories_y': 'categories', 'category_sort_y': 'category_sort',
                              'jan': 'jan', 'feb': 'feb', 'mar': 'mar', 'apr': 'apr', 'may': 'may',
                              'jun': 'jun', 'jul': 'jul', 'aug': 'aug', 'sep': 'sep', 'oct': 'oct',
                              'nov': 'nov', 'dec': 'dec', 'tot': 'tot'}, inplace=True)
    result_df = pd.concat([grouped_df,df[mask_starts_with_zero]], ignore_index=True)
    

    return result_df
                
def budget_status(df,post_df,v,output_method=0):
    if isinstance(output_method,list):
        output_method_num = output_method[0]
        output_method_mon = get_months(output_method[1])
        df_temp,v = create_pages(df,v,sel=output_method_num)
    filters_ini = {"key": "", "type": "posts"}
    # Loop over each row in the DataFrame and update budget values
    _month_col = get_months()
    for index, row in df.iterrows():
        year = int(row['year'])
        if year == datetime.now().year:
            cur_mon = datetime.now().month
            _month_col = get_months(range(1,cur_mon+1))
        for month in _month_col:
            _mon = get_months(month)
            filt = filters_ini.copy()
            date1 = datetime(year,_mon,1)
            _, last_day = calendar.monthrange(year, _mon)
            date2 = datetime(year,_mon,last_day)
            filt['date'] = {'min': date1,'max':date2}
            if not v['seperate_marked_in_budget'] or row['category'] != 'Marked':
                cat = []
                for _typ,_cat in zip(row['category_type'].split(','),row['category'].split(',')):
                    cat.append(str(_typ) + _cat)
                filt['category'] = {'equal': cat, 'not_empty':True}
                if v['seperate_marked_in_budget']:
                    filt['marked'] = {'empty': True, 'not_empty':False}
            else:
                filt['marked'] = {'equal': [row['name']], 'not_empty':True}
            filter_post_df = filter_dataframe(post_df, filt)
            if isinstance(output_method,list):
                if month in output_method_mon:
                    if row['id'] == df_temp['id']:
                        print(f'Month {month}, {len(filter_post_df)} posts')
                        if len(filter_post_df):
                            print(filter_post_df[['date','text','amount','category_type','category']])
            post_sum = filter_post_df['amount'].sum()
            df.at[index, month] = post_sum - df.at[index, month]
    df['tot'] = df[_month_col].sum(axis=1)

    return df 

def set_view_and_filter(_input_key,_settings,_view,_df_filter={}):
    global warn_message
    global skip_update
    if 'filters' in _settings:
        for fil,i in zip(_settings['filters'],range(len(_settings['filters']))):
            if 'key' in fil:
                skip_update = False
                if _input_key == fil['key']:
                    _df_filter = fil.copy()
                    break
    if 'views' in _settings:
        for vie,i in zip(_settings['views'],range(len(_settings['views']))):
            if 'key' in vie:
                if _input_key == vie['key']:
                    skip_update = False
                    _view["type"] = vie['type']
                    _view["view"] = i
                    _view["page"] = '1'
                    if 'group' in vie: 
                        _view["group"] = vie['group']
                    else:
                        _view["group"] = None
                    break
    if len(_df_filter) and _view["type"] != _df_filter["type"]:
        _df_filter = {}
    if _view["type"][:min(6,len(_view["type"]))] == 'budget':
        if _view["year"] == 0:
            _view["year"] = datetime.now().year
        if _view["year"] in budgets:
            skip_update = False
            _cur_df = budgets[_view["year"]].copy()
            if _view["group"] == 'category':
                _cur_df = group_and_sum_rows(_cur_df,1)
                '''
                if view["type"] == 'budget status':
                    post_dfs = dfs_history[current_history_index].copy()
                    cur_df = budget_status(cur_df,post_dfs,view)
                '''
        else:              
            warn_message.append(f'Budget {view["year"]} not found')
            _cur_df = None
            skip_update = True
    elif len(dfs_history) and _view['type'] == 'posts':
        _cur_df = dfs_history[current_history_index].copy()
    else:
        _cur_df = None
    return _cur_df,_df_filter,_view

def set_filter(filter_input,existing_filter={}):
    if len(filter_input) == 1: 
        if filter_input[0][:min(3,len(filter_input[0]))] == 'res':
            existing_filter = settings['filters'][view['filter']].copy()
    if len(filter_input) == 2: 
        if filter_input[0] == 'del':
            _column = find_column(filter_input[1])
            if _column in existing_filter:
                del existing_filter[_column]
        elif filter_input[0] in ['sor','sor2','sort','sort2']:
            if filter_input[0][-1] != '2':
                sort_name = 'sort'
            else:
                sort_name = 'sort2'
            if filter_input[1] == 'asc' or filter_input[1] == 'des':
                if filter_input[0][-1] != '2':
                    existing_filter[sort_name]['direction'] = filter_input[1]
                else:
                    existing_filter[sort_name]['direction'] = filter_input[1]
            else:
                _column = find_column(filter_input[1])
                if _column is not None:
                   if existing_filter[sort_name]['column'] != _column:
                       existing_filter[sort_name]['column'] = _column
                   else:
                       if existing_filter[sort_name]['direction'] == 'des':
                           existing_filter[sort_name]['direction'] = 'asc'
                       else:
                           existing_filter[sort_name]['direction'] = 'des'
        else:
            _column = find_column(filter_input[0])
            existing_filter = change_filter(existing_filter,_column,filter_input[1])
    return existing_filter

def set_plot(_input_key,_settings):
    global warn_message
    if 'plots' in settings:
        for plo,i in zip(_settings['plots'],range(len(_settings['plots']))):
            if 'key' in plo:
                if _input_key == plo['key']:
                    return plo
    return {}

def evaluate_textstring(template, variable_dict):
    """
    Replace placeholders in the template string with variable values.

    Args:
    - template (str): The template string with placeholders.
    - variable_dict (dict): A dictionary containing variable names and values.

    Returns:
    - str: The formatted string.
    """
    # Use eval to evaluate expressions within curly braces
    def eval_expression(match):
        expression = match.group(1)
        try:
            #return str(eval(expression, variable_dict))
            return f"{eval(expression, variable_dict):,.0f}"
        except Exception as e:
            print(f"Error evaluating expression '{expression}': {e}")
            return match.group(0)

    # Use regular expressions to find expressions within curly braces
    import re
    pattern = re.compile(r'\{([^}]+)\}')

    # Replace placeholders with evaluated expressions
    formatted_string = pattern.sub(eval_expression, template)
    return formatted_string

def undo_redo_dataframe(user_input):
    global current_history_index
    global warn_message
    if view["type"] == 'posts':
        if user_input[0].lower() == 'u':
            current_history_index += 1
        elif user_input[0].lower() == 'r':
             current_history_index -= 1
        if current_history_index < 0:
            warn_message.append('Redo not possible. Current stage is the latest')
            current_history_index = 0
        elif current_history_index >= len(dfs_history):
            warn_message.append('Undo not possible. No more history saved')
            current_history_index = len(dfs_history) - 1
        cur_df = dfs_history[current_history_index].copy()
        return cur_df

def clear_dataframe():
    print('Are you sure to clear all data?')
    clear_input = input('y/n: ')
    if clear_input == 'y':
        dfs_history = []
        cur_df = None
        current_history_index = 0
        print('Ok')
        return dfs_history,cur_df,current_history_index

def save_load_dataframe(_type,pkl_path):
    global skip_update
    global warn_message
    if _type == 'w':
        if os.path.exists(pkl_path):
            print(f'Do you want to overwrite {pkl_path}?')
            overwrite_input = input('y/n: ')
        else:
            overwrite_input = 'y'
        if not overwrite_input == 'y':
            pkl_path = filedialog.asksaveasfilename(defaultextension=".pkl",filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],title="Save As")
    if _type == 'l':
        if len(pkl_path) == 0 or not os.path.exists(pkl_path):
            pkl_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
            if len(pkl_path) == 0 or not os.path.exists(pkl_path):
                skip_update = True
                warn_message.append('Cancelling load')
        if len(pkl_path) > 0:
            if os.path.exists(pkl_path):
                if len(dfs_history) > 0:
                    print(f'Do you want to load {os.path.abspath(pkl_path)}?')
                    load_input = input('Unsaved data will be deleted. y/n/c: ')
                else:
                    load_input = 'y'
                if load_input == 'y':
                    cur_df = pd.read_pickle(pkl_path)
                    print(f'Loading dataframe from {os.path.abspath(pkl_path)}')
                    pickle_file_path = pkl_path
                    return cur_df
                elif load_input == 'n':
                    pkl_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
                    if len(pkl_path) == 0 or not os.path.exists(pkl_path):
                        skip_update = True
                        warn_message.append('Cancelling load')
                    else:
                        cur_df = pd.read_pickle(pkl_path)
                        print(f'Loading dataframe from {os.path.abspath(pkl_path)}')
                        pickle_file_path = pkl_path
                        return cur_df
                else:
                    skip_update = True
                    warn_message.append('Cancelling load')
            else:
                skip_update = True
                warn_message.append('Cancelling load')
    elif _type == 'w':
        dfs_history[current_history_index].to_pickle(pkl_path)
        print(f'Writing dataframe to {os.path.abspath(pkl_path)}')
        pickle_file_path = pkl_path
        print(f'ok')
        skip_update = True

def export_dataframe():
    if 'export' in settings:
        for output,item in settings['export'].items():
            export_text = []
            variables = {}
            for _item in item:
                _type = _item.split(':')[0]
                _input = _item[len(_type)+1:]
                if _type == 'text':
                    _string = evaluate_textstring(_input, variables)
                    export_text.append(_string)
                    print(_string)
                else: 
                    _input_split = _input.split(';')
                    par_type = _input_split[0]
                    export_view = view
                    export_view['type'] = par_type
                    export_view['year'] = datetime.now().year
                    name = ''
                    _id = ''
                    value = ''
                    return_value = -99999999999
                    export_filter = {}
                    for _par in _input_split[1:]:
                        par_input_type = _par.split('=')[0]
                        par_input_input = _par.split('=')[1]
                        if par_input_type == 'key':
                            export_df,export_filter,export_view = set_view_and_filter(par_input_input,settings,export_view)
                        elif par_input_type == 'name':
                            name = par_input_input
                        elif par_input_type == 'id':
                            _id = par_input_input
                        elif par_input_type == 'value':
                            value = par_input_input
                        elif par_input_type == 'year':
                            export_view["year"] = int(par_input_input)
                        elif par_input_type == 'filter':
                            par_split = shlex.split(par_input_input)
                            export_filter = set_filter(par_split,export_filter)
                    if _type != 'df':
                        export_df = create_output(export_df,export_filter,export_view,'df')
                    else:
                        table_df = create_output(export_df,export_filter,export_view,'table')
                    if par_type == 'budget':
                        if 'current_month' == value:
                            _column = get_months(datetime.now().month)
                            if len(name) > 0:
                                return_value = export_df.loc[export_df['name'] == name, _column].values[0]
                            if len(_id) > 0:
                                return_value = export_df.loc[export_df['id'] == _id, _column].values[0]
                        elif 'current_month_sum' == value:
                            _column_sum = get_months(range(1,datetime.now().month+1))
                            if len(name) > 0:
                                return_value = export_df.loc[export_df['name'] == name, _column_sum].sum(axis=1).values[0]
                            if len(_id) > 0:
                                return_value = export_df.loc[export_df['id'] == _id, _column_sum].sum(axis=1).values[0]
                        else:
                            _column_sum = value
                            if len(name) > 0:
                                return_value = export_df.loc[export_df['name'] == name, _column_sum].sum(axis=1).values[0]
                            if len(_id) > 0:
                                return_value = export_df.loc[export_df['id'] == _id, _column_sum].sum(axis=1).values[0]
                    elif par_type == 'posts':
                        if 'sum' in value:
                            return_value = export_df[value.split('.')[0]].sum()
                    if _type == 'df':
                        pretty_table_string = table_df.get_string().splitlines()
                        export_text.extend(pretty_table_string)
                    if par_type != 'df':
                        variables[_type] = return_value
            for output_item in output.split(','):
                if '@' in output_item:
                    import smtplib
                    import ssl
                    import html
                    from email.mime.text import MIMEText
                    from email.mime.multipart import MIMEMultipart
                    # Sender's email address and password
                    sender_email = settings['email']['email']
                    password = settings['email']['app_password']
                    if len(password) != 16:
                        print('Email app password needs to be a 16 character key')
                    
                    # Create the MIME object
                    message = MIMEMultipart()
                    message["From"] = sender_email
                    message["To"] = output_item
                    message["Subject"] = export_text[0]
                    
                    # Add body to the email
                    # Generate the email body
                    body = "<html><body style='font-family: \"Courier New\", monospace;'>"
                    
                    for element in export_text[1:]:
                        if isinstance(element, str):
                            body += f"<pre>{html.escape(str(element))}</pre>"
                        else:
                            # Handle other cases if needed
                            pass
                    
                    body += "</body></html>"
                    message.attach(MIMEText(body, "html"))
                    
                    # Establish a secure connection with the SMTP server
                    context = ssl.create_default_context()
                    
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                        server.login(sender_email, password)
                        # Send the email
                        server.sendmail(sender_email, output_item, message.as_string())
                else:
                    with open(output_item, 'w') as file:
                        for string_item in export_text:
                            file.write(f"{string_item}\n")

def show_detail(_input):
    global skip_update
    if view['type'] == 'posts':
        change_ids = create_output(cur_df,df_filters,view,_input[0])
        print(len(change_ids))
        if len(change_ids) == 1:
            print_df = cur_df.loc[cur_df['id'].isin(change_ids)]
            print()
            print('***Imported/fixed data columns')
            print(f'id: ' + str(print_df['id'].values[0]))
            print(f'posted_date: ' + str(print_df['posted_date'].dt.strftime(settings['date_format_print']).values[0]))
            print(f'text: ' + print_df['text'].values[0])
            print(f'value: ' + str(print_df['value'].values[0]))
            print(f'balance: ' + str(print_df['balance'].values[0]))
            print(f'account: ' + print_df['account'].values[0])
            print(f'info: ' + print_df['info'].values[0])
            print(f'imported: ' + str(print_df['imported'].dt.strftime(settings['date_format_print'] + ' %H:%M:%S').values[0]))
            print('***Category/changable data')
            print(f'date: ' + str(print_df['date'].dt.strftime(settings['date_format_print']).values[0]))
            cat_typ = print_df['category_type'].values[0]
            cat = print_df['category'].values[0]
            print(f'category_type: ' + str(cat_typ))
            print(f'category: ' + cat)
            if cat_typ != '':
                category_name = settings['categories'][cat_typ]['name']
                sub_category_name = settings['categories'][cat_typ][cat]
                print(f'Category display name: {category_name} - {sub_category_name}')
            print(f'amount: ' + str(print_df['amount'].values[0]))
            print(f'split_id: ' + str(print_df['split_id'].values[0]))
            print(f'marked: ' + print_df['marked'].values[0])
            print(f'notes: ' + print_df['notes'].values[0])
            print(f'status: ' + print_df['status'].values[0])
            print(f'changed: ' + str(print_df['changed'].dt.strftime(settings['date_format_print'] + ' %H:%M:%S').values[0]))
        else:
            print('Only 1 selection allowed')
    elif view['type'] == 'budget status':
        change_ids = create_output(cur_df,df_filters,view,_input)
        #skip_update = True
    skip_update = True

def change_dataframe_handle(_input,df,df_filt={},vi=[]):
    global dataframe_update
    global skip_update
    _column = find_column(_input[1],False)
    if _column is not None:
        if len(_input[0]) > 3 and _input[0][:3] == 'id=':
            change_ids = pd.Series([int(_input[0][3:])])
        else:
            change_ids = create_output(df,df_filt,vi,_input[0])
        if len(change_ids) > 0:
            _value = _input[2]
            if _value != '':
                _value = convert_value_to_data_type(df,_column,_input[2])
            if _value is not None:
                if _column == 'category':
                    if _value != '':
                        _value = autocomplete_category(_value)
                    else:
                        _value = ['','']
                    if _value is not None:
                        dataframe_update = True
                        df = change_dataframe_rows(df,_column,change_ids,_value[1])
                        df = change_dataframe_rows(df,'category_type',change_ids,_value[0])
                    else: skip_update = True
                elif _column == 'marked':
                    if _value != '':
                        _value = autocomplete_marked(_value)[0]
                    if _value is not None:
                        dataframe_update = True
                        df = change_dataframe_rows(df,_column,change_ids,_value)
                    else: skip_update = True
                else:
                    dataframe_update = True
                    df = change_dataframe_rows(df,_column,change_ids,_value)
            else: skip_update = True
    else: skip_update = True
    return df

def drop_dataframe_handle(_input,df,df_filt={},vi=[]):
    global dataframe_update
    global skip_update
    change_ids = create_output(df,df_filt,vi,_input[0])
    if len(change_ids):
        df = df[~df['id'].isin(change_ids)]
        skip_update = False
        dataframe_update = True
    return df

def split_dataframe_handle(_input,df,df_filt,vi):
    global dataframe_update
    global skip_update
    if _input[0] == 'del':
        change_ids = create_output(df,df_filt,vi,_input[1])
        if len(change_ids) > 0:
            df = delete_split_dataframe_rows(df, change_ids)
            dataframe_update = True
        else: skip_update = True
    else:
        change_ids = create_output(df,df_filt,vi,_input[0])
        if len(change_ids) > 0:
            try:
                new_value = float(_input[1])
            except:
                new_value = _input[1]
            if new_value is not None and len(str(new_value)) > 0:
                if len(_input) > 2 and len(_input[2]) > 0: #Create new_category
                    if _input[2] != '-':
                        new_cat = autocomplete_category(_input[2])
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
                                new_cat = autocomplete_category(_cat_input)
                                if not new_cat == None:
                                    no_cat = False
                            elif _cat_q[0].lower() == 'c':
                                new_cat = None
                                no_cat = False
                                print('Skipping command')
                            elif _cat_q[0].lower() == 'n':
                                new_cat = '-'
                                no_cat = False
                    else:
                        new_cat = '-' #Split to same category
                else:
                    new_cat = '' #
                if len(_input) > 3 and len(_input[3]) > 0: #Create new_mark 
                    if _input[3] == '-':
                        new_mark = '-'
                    else:
                        new_mark = autocomplete_marked(_input[3])
                else:
                    new_mark = ''
                if len(_input) > 4: #Create note 
                    if _input[4][0] == 'y':
                        print('Enter new note')
                        new_note = input('')
                else:
                    new_note = None
                dataframe_update = True
                df = split_dataframe_rows(df, change_ids, new_value, new_cat, new_mark, new_note)
            else: skip_update = True
        else: skip_update = True
    return df

def handle_user_input(user_input,df,df_filt,vi):
    global plot_df
    global dataframe_update
    global skip_update
    global warn_message
    global current_history_index
    global clear_df
    # Undo and redo dataframe changes
    if user_input[0] in ['u','r']:
        df = undo_redo_dataframe(user_input)
    # Clear dataframe
    elif user_input[0] == 'clear':
        clear_df = True
        #clear_dataframe()
    # Save and load
    elif user_input[0] in ['w','l']:
        df = save_load_dataframe(user_input[0],pickle_file_path)
        if cur_df is not None:
            dataframe_update = True
    # import dataframe
    elif user_input[0] == 'i':
        dataframe_update = True
        df = import_data(df)
        if df is None:
            skip_update = True
            warn_message.append('Import canceled')
    elif user_input[0] == 'ic':
        df = import_change_by_id(df)
        if df is None:
            skip_update = True
        else:
            dataframe_update = True
    # export data
    elif user_input[0] == 'e':
        export_dataframe()
    # Print help text
    elif user_input[0] == 'h':
        print_help('basic',view)
        skip_update = True
    # Change page
    elif user_input[0] in ['pn','pp','p']:
        if len(user_input) == 1 and user_input[0] == 'p':
            print_help('page',view)
        else:
            if len(user_input) > 1:
                add = int(user_input[1])
            elif user_input[0] == 'pn':
                add = 1
            elif user_input[0] == 'pp':
                add = -1
            if user_input[0] == 'p':
                vi["page"] = str(add)
            else:
                vi["page"] = str(int(vi["page"]) + add)
    # Change budget year
    elif user_input[0] in ['bn','bp']:
        if _input == 'bn' or _input == 'bp':
            if _input == 'bn':
                add = 1
            else:
                add = -1
            view["year"] = int(view["year"]) + add
    # Change filter
    elif user_input[0] == 'f':
        if len(user_input) == 1:
            print_help('filter',view)
        else:
            df_filt = set_filter(user_input[1:],df_filt)
    # Show detail
    elif user_input[0] == 'd':
       if len(user_input) == 1:
           print_help('detail',view)
       else:
           show_detail(user_input[1:])
    elif user_input[0] == 'ovs':
        df_filtered = filter_dataframe(df, df_filt)
        overview(df_filtered, mode='simple')
        skip_update = True
    elif user_input[0] == 'ovm':
        df_filtered = filter_dataframe(df, df_filt)
        overview(df_filtered, mode='months')
        skip_update = True
    # Plot dataframe
    # Overview plot
    elif user_input[0] == 'plo':
        plot_df = {'type':'overview','df_filter':df_filt,'plot_options':{}}
        skip_update = True
    # Sankey diagram
    elif user_input[0] == 'pls':
        plot_df = {'type':'sankey','df_filter':df_filt,'plot_options':{'monthly':True}}
        skip_update = True
    # Change dataframe
    elif user_input[0] == 'c':
        if len(user_input) == 1:
            print_help('change',view)
        elif len(user_input) >= 4:
            df = change_dataframe_handle(user_input[1:],df,df_filt,vi)
    # Change dataframe
    elif user_input[0] == 'del':
        if len(user_input) == 1:
            print_help('delete',view)
        else:
            print('Deleting single rows in the dataframe is not recommended. Only entire accounts')
            con = input('Do you want to continue deleting y/n?')
            if con == 'y':
                df = drop_dataframe_handle(user_input[1:],df,df_filt,vi)
            else:
                print('Skipping delete command')
    # Auto change dataframe
    elif user_input[0] == 'auto':
        if len(user_input) > 1:
            id_sel = user_input[1]
        else:
            id_sel = 'all'
        change_ids = create_output(df,df_filt,vi,id_sel,silent=True)
        df = auto_change_dataframe(df,df_filt, vi,change_ids)
        dataframe_update = True
    # Calc loans
    elif user_input[0] == 'dept':
        loan_parameters = substitute_parameters(df,vi,settings['dept'])
        annuity_table = calculate_annuity_table(loan_parameters)
        print_annuity_table(annuity_table)
        skip_update = True
    # Split dataframe
    elif user_input[0] == 's':
        if len(user_input) == 1:
            print_help('split',view)
        elif len(user_input) >= 3: 
            df = split_dataframe_handle(user_input[1:],df,df_filt,vi)
    elif user_input[0] == 'views':
        skip_update = True
        if len(settings['views']):
            print()
            print("key: view name")
        for v in settings['views']:
            print(f"{v['key']}: {v['name']}")
    elif user_input[0] == 'plots':
        skip_update = True
        if 'plots' in settings:
            if len(settings['plots']):
                print()
                print("key: plot name")
                for p in settings['plots']:
                    print(f"{p['key']}: {p['name']}")
            else:
                print('No plots defined in setting.plots')
        else:
            print('No setting.plots file was found')
    return df,df_filt,vi

def check_and_correct_df(df):
    global dataframe_update
    if df is not None:
        # Check if 'split_id' column exists in the DataFrame
        if 'split_id' not in df.columns:
            print("Error: 'split_id' column not found in the DataFrame.")
            return df
        int_columns = ['id','split_id']
        for col in int_columns:
            # Check the data type of int column
            if df[col].dtype == 'object' or df[col].dtype == 'float64':
                # Convert 'split_id' column to integer
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')
                    print(f"'{col}' column converted to integers.")
                    dataframe_update = True
                except ValueError as e:
                    print(f"Error converting '{col}' column to integers: {e}")
            elif df[col].dtype == 'int64':
                Skip = True
            else:
                print(f"'{col}' column has an unexpected data type:", df[col].dtype)

        # Convert columns to datetime if they are not already
        date_columns = ['posted_date', 'date', 'changed', 'imported']
        for col in date_columns:
            if df[col].dtype != 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col])
        # Check for rows where 'value' is 0 and 'split_id' is not -1
        zero_value_minus_one_rows = df[(df['value'] == 0) & (df['split_id'] == -1)]
        
        # Check and fix rows with the same values in specific columns for the entire DataFrame
        for _, group in df[df.duplicated(subset=['posted_date', 'balance', 'text', 'info'], keep=False)].groupby(['posted_date', 'balance', 'text', 'info']):
            if any(group['id'].isin(zero_value_minus_one_rows['id'])):
                max_id_row = group[group['id'] == group['id'].max()]
                min_id_row = group[group['id'] == group['id'].min()]
        
                if max_id_row['value'].iloc[0] == 0 and min_id_row['value'].iloc[0] != 0:
                    print(f"Changing values for row with id {max_id_row['id'].iloc[0]} and row with id {min_id_row['id'].iloc[0]}")
                    max_abs_value_id = group.loc[group['value'].abs().idxmax(), 'id']
                    group.loc[group['id'] == group['id'].min(), 'value'] = group.loc[max_abs_value_id, 'value']
        
                    # Set 'split_id' for the row with higher id to the id of the row with lower id
                    group.loc[group['id'] != group['id'].min(), 'split_id'] = int(min_id_row['id'].iloc[0])
                    df.update(group)
                    dataframe_update = True


        # Check for NaN values in the entire DataFrame
        nan_values = df.isnull().sum().sum()
        if nan_values > 0:
            print(f"Warning: The DataFrame contains {nan_values} NaN values.")
            nan_rows = df[df.isnull().any(axis=1)]
            print("Rows with NaN values:")
            print(nan_rows[['id'] + list(df.columns[df.isnull().any()])])

        '''
        # Check for rows where 'value' is 0 and 'split_id' is not -1
        zero_value_non_minus_one_rows = df[(df['value'] == 0) & (df['split_id'] != -1)]
        
        # Check and fix rows with the same values in specific columns for the identified rows
        for _, group in zero_value_non_minus_one_rows.groupby(['posted_date', 'balance', 'text', 'info']):
            if group['value'].nunique() == 1 and group['value'].iloc[0] == 0:
                max_id_row = group[group['id'] == group['id'].max()]
                min_id_row = group[group['id'] == group['id'].min()]
        
                if max_id_row['value'].iloc[0] == 0 and min_id_row['value'].iloc[0] != 0:
                    print(f"Changing values for row with id {max_id_row['id'].iloc[0]} and row with id {min_id_row['id'].iloc[0]}")
        
                    # Swap 'value' for rows with the same values
                    df.loc[df['id'] == max_id_row['id'].iloc[0], 'value'] = min_id_row['value'].iloc[0]
                    df.loc[df['id'] == min_id_row['id'].iloc[0], 'value'] = max_id_row['value'].iloc[0]
        
                    # Set 'split_id' for the row with higher id to the id of the row with lower id
                    df.loc[df['id'] == max_id_row['id'].iloc[0], 'split_id'] = min_id_row['id'].iloc[0]


        duplicate_rows = df[df.duplicated(subset=['posted_date', 'balance', 'text', 'info'], keep=False)]
        for _, group in duplicate_rows.groupby(['posted_date', 'balance', 'text', 'info']):
            if group['value'].nunique() == 1 and group['value'].iloc[0] != 0:
                max_id_row = group[group['id'] == group['id'].max()]
                min_id_row = group[group['id'] == group['id'].min()]
        
                if min_id_row['value'].iloc[0] == 0 or min_id_row['split_id'].iloc[0] != -1:
                    print(f"Changing values for row with id {max_id_row['id'].iloc[0]} and row with id {min_id_row['id'].iloc[0]}")
                    # Swap 'value' for rows with the same values
                    df.loc[df['id'] == max_id_row['id'].iloc[0], 'value'] = min_id_row['value'].iloc[0]
                    df.loc[df['id'] == min_id_row['id'].iloc[0], 'value'] = max_id_row['value'].iloc[0]
        
                    # Set 'split_id' for the row with higher id to the id of the row with lower id
                    df.loc[df['id'] == min_id_row['id'].iloc[0], 'split_id'] = -1
                    df.loc[df['id'] == max_id_row['id'].iloc[0], 'split_id'] = min_id_row['id'].iloc[0]

        # Check for rows where ['value'] == 0 and ['split'] == -1
        zero_value_split_minus_one_rows = df[(df['value'] == 0) & (df['split_id'] == -1)]
        if not zero_value_split_minus_one_rows.empty:
            print("Warning: Rows where ['value'] == -1 and ['split_id'] == -1:")
            print(zero_value_split_minus_one_rows[['id', 'text']])
        '''
    return df

def calculate_annuity_table(loan_parameters):
    """
    Calculate the annuity table based on loan parameters.
    """
    annuity_tables = []

    for loan in loan_parameters:
        name = loan['name'][5:]
        annuity_table = []
        dept = loan['dept']  # Current dept
        date = loan['date'].replace(day=1)  # Set day to 1st of the month to consider only month and year
        yearly_payments = loan['yearly_payments']
        interest_rate = loan['interest_rate']  # Interest_rate in fraction
        processing_fee = loan['processing_fee_value']  # Processing fee value at each payment
        processing_fee_rate = loan['processing_fee_rate']  # Yearly processing fee rate at each payment
        processing_fee_rate_value = loan['processing_fee_rate_value']
        annuity_payment = loan['annuity_payment'] # Specified annuity payment
        total_payment = loan['payment']
        installment_payment = loan['installment']
        interest_payment = loan['interest_value']

        # Calculate annuity interest rate
        r = interest_rate / yearly_payments
        annuity_table.append({
            'date': date,
            'dept': dept,
            'payment': total_payment,
            'installment': installment_payment,
            'interest': interest_payment,
            'processing_fee': processing_fee,
            'processing_fee_rate': processing_fee_rate_value,  # Adjust processing fee for remaining debt
            'total_fee': processing_fee_rate_value + processing_fee
        })
        new_year = date.year 
        while dept > 0 and len(annuity_table) < 1200:
            interest_payment = dept * r  # Interest payment for this period
            installment_payment = annuity_payment - interest_payment  # Principal payment for this period
            # Ensure the last payment doesn't exceed the remaining debt
            if installment_payment > dept:
                installment_payment = dept

            processing_fee_rate_value = dept * processing_fee_rate / yearly_payments
            total_fee = processing_fee_rate_value + processing_fee
            # Add processing fee to the payment
            total_payment = annuity_payment + total_fee


            dept -= installment_payment
            # Advance to the next month
            new_month = date.month + int(12/yearly_payments)
            if new_month > 12:
                new_year = date.year + int((new_month-1)/12) 
                new_month = new_month - int(new_month/13)*12
            date = datetime(new_year, new_month, 1)
            #date = datetime(date.year + (date.month + 1) // 12, ((date.month + 1) % 12) + 1, 1)  # Move to next month, considering year rollover

            # Append payment details to the annuity table
            annuity_table.append({
                'date': date,
                'dept': dept,
                'payment': total_payment,
                'installment': installment_payment,
                'interest': interest_payment,
                'processing_fee': processing_fee,
                'processing_fee_rate': processing_fee_rate_value,
                'total_fee': total_fee
            })
        annuity_tables.append({'name':name,'parameters':loan,'table':pd.DataFrame(annuity_table)})


    return annuity_tables

def print_annuity_table(annuity_tables):
    table = pt()
    table.field_names = ["Name", "Dept", "Interest %", "Next Payment", "Payment","Interest Value", "Installment", "Fee","Paid out"]
    _table_align = ["l","r","r","c","r","r","r","r","c"]
    
    for entry in annuity_tables:
        name = entry['name']
        parameters = entry['parameters']
        table_df = entry['table']
        dept = "{:,.0f}".format(table_df.iloc[0]['dept']) if len(table_df) > 1 else "N/A"
        interest = "{:.2f}".format(parameters['interest_rate']*100) if 'interest_rate' in parameters else "N/A"
        paid_out = table_df.iloc[-1]['date'].strftime("%Y-%m") if len(table_df) > 1 else "N/A"
        next_payment = table_df.iloc[1] if len(table_df) > 1 else None
        if next_payment is not None:
            np_date = next_payment['date'].strftime("%Y-%m")
            np_amount = "{:,.0f}".format(next_payment['payment'])
            np_interest = "{:,.0f}".format(next_payment['interest'])
            np_installment = "{:,.0f}".format(next_payment['installment'])
            np_fee = "{:,.0f}".format(next_payment['total_fee'])
        else:
            np_date = "N/A"
            np_amount = "N/A"
            np_interest = "N/A"
            np_installment = "N/A"
            np_fee = "N/A"
        table.add_row([
            name,
            dept,
            interest,
            np_date,
            np_amount,
            np_interest,
            np_installment,
            np_fee,
            paid_out
        ])
    for col, align in zip(table.field_names,_table_align):
        table.align[col] = align
    
    print(table)

def calc_parameters(key,value,df,substituted_loan,mode=''):
    if '{' in value and '}' in value:
        if mode == '':
            for param in re.findall(r'{(.*?)}', value):
                if param not in substituted_loan:
                    substituted_loan = calc_missing_parameters(param,df,substituted_loan)
                value = value.replace(f'{{{param}}}', str(substituted_loan[param]))
            substituted_loan[key] = float(eval(value))
    elif '[d]' in value or '[dd]' in value:
        # Handle [d] or [dd] for values after ':'
        if mode == '':
            search_string = df.iloc[-1]['info'] 
        elif mode == '_last':
            search_string = df.iloc[-2]['info'] 
        if ':::' in value:
            funcs = value.split(':::')[:-1]
            value = value.split(':::',1)[-1]
            for func in funcs:
                if func[:5] == 'split':
                    split_val = func.split('(')[1].split(')')[0]
                    if ',' in split_val:
                        split_val1 = split_val.split(',')[0]
                        split_val2 = split_val.split(',')[1]
                    else:
                        split_val1 = split_val
                        split_val2 = ''
                    search_string = search_string.split(split_val1,1)[1]
                    if len(split_val2): 
                        search_string = search_string.split(split_val2,1)[0]
        if '[dd]' in value:
            val2 = value.replace('[d]','').replace('[dd]','').strip()
            search_string =  search_string.strip().replace(val2,'!value_code!')
            search_string = search_string.replace('.','!pre!').replace(',','.').replace('!pre!',',')
            search_string = search_string.replace('!value_code!',val2)
        # Define the placeholder patterns
        #placeholder_patterns = {'[d]': r'([\d,]+(\.\d*)?)', '[dd]': r'([\d,]+(\.\d*)?)'}
        #placeholder_patterns = {' [d]': r'\s+([\d,]+(\.\d*)?)', '[d]': r'([\d,]+(\.\d*)?)'}
        #                                  \s+([\d,]+(?:\.\d*[1-9])?)
        #placeholder_patterns = {' [d]': r'\s+([\d,]+(?:\.\d*[1-9])?)', '[d]': r'([\d,]+(\.\d*)?)'}
        placeholder_patterns = {' [d]': r'\s+([\d,]+(?:,\d{3})*(?:\.\d*)?)', '[d]': r'([\d,]+(\.\d*)?)'}
        #                                  \s+([\d,]+(?:,\d{3})*(?:\.\d*)?)'
        
        # Define your original pattern
        pattern = value.replace('[dd]','[d]')

        n_res = pattern.count('[d]')
        
        # Replace placeholders with their corresponding regular expression patterns
        for placeholder, regex_pattern in placeholder_patterns.items():
            pattern = pattern.replace(placeholder, regex_pattern)

        #value2 = value.replace(' [d]', r'\s+([\d,]+(\.\d*)?)').replace(' [dd]',r'\s+([\d,]+(\.\d*)?)')
        #digits_match = re.search( value2,search_string)
        digits_match = re.search(pattern,search_string)
        if digits_match != None:
            substituted_loan[key+mode] = float(digits_match.group(n_res).replace(',',''))
        else:
            if key+mode in substituted_loan:
                del substituted_loan[key+mode]
    elif value.isdigit() and  mode == '':
        substituted_loan[key] = float(value)
    elif mode == '':
        print(f"Error for key {key}")
    return substituted_loan
 
def substitute_parameters(df,view,loan_parameters):
    """
    Substitute parameters enclosed in curly braces {} with their corresponding values.
    """
    substituted_parameters = []
    def_filter = {"key": "", "type": "posts","sort": {"column": "date", "direction": "des"},"sort2": {"column": "id", "direction": "des"}}

    for loan in loan_parameters:
        substituted_loan = loan.copy()
        substituted_loan = set_default_parameters(substituted_loan)
        df_filt = def_filter.copy()
        split_characters = ['>', '<', '=']
        for filt in loan['filter']:
            column = find_column(re.split(f'[{"".join(split_characters)}]', filt)[0])
            split_character = next(char for char in filt if char in split_characters)
            value = filt.split(split_character)[1].strip()
            if column in ['text','info','category_id','account','category','status','marked','notes']:
                incl_sc = ''
            else:
                incl_sc = split_character
            df_filt = set_filter([column,incl_sc + value],df_filt)
        _df = create_output(df,df_filt,view,'df')
        if len(df) == 0:
            break
        if 'yearly_payments' not in loan:
            if len(_df) > 1:
                # Calculate the difference in months
                date1 = _df.iloc[-1]['date']
                date2 = _df.iloc[-2]['date']
                
                # Calculate difference in months
                months_diff = (date1.year - date2.year) * 12 + (date1.month - date2.month) 
                substituted_loan['yearly_payments'] = int(round(12/months_diff))
            else:
                substituted_loan['yearly_payments'] = 12
        # Create df from filter
        if 'payment' not in loan:
            substituted_loan['payment'] = _df.iloc[-1]['amount']*(-1)
            if len(_df) > 1:
                if 'payment_last' not in loan:
                    substituted_loan['payment_last'] = _df.iloc[-2]['amount']*(-1)
        if 'date' not in loan:
            substituted_loan['date'] = _df.iloc[-1]['date']
            if len(_df) > 1:
                if 'date_last' not in loan:
                    substituted_loan['date_last'] = _df.iloc[-2]['date']
        for loop in range(2):
            if len(_df) > 1:
                for key, value in loan.items():
                    if key not in ['name','filter']:
                        if key + '_last' not in loan:
                            substituted_loan = calc_parameters(key,value,_df,substituted_loan,'_last')
            for key, value in loan.items():
                if key not in ['name','filter']:
                    search_string = _df.iloc[-1]['info'] 
                    substituted_loan = calc_parameters(key,value,_df,substituted_loan)
        if 'processing_fee_value' not in loan:
            substituted_loan['processing_fee_value'] = 0.0
        changed = 100
        while changed > 0:
            old_len = len(substituted_loan)
            substituted_loan = calc_missing_parameters('all',_df,substituted_loan)
            new_len = len(substituted_loan)
            changed = new_len - old_len


        substituted_parameters.append(substituted_loan)

    return substituted_parameters


def calc_missing_parameters(calc_val,df,sl):
    def calc_interest_rate(sl,mode=''):
        if 'interest_rate' not in sl:
            if all(string in sl for string in ['interest_rate_p']):
                sl['interest_rate'] = sl['interest_rate_p']/100
            elif all(string in sl for string in ['interest_value'+mode,'dept_last'+mode,'yearly_payments']):
                sl['interest_rate'] = sl['yearly_payments'] * sl['interest_value'+mode] / sl['dept_last'+mode]
        return sl
    def calc_interest_value(sl,mode=''):
        if 'interest_value'+mode not in sl:
            if all(string in sl for string in ['interest_rate','dept_last'+mode,'yearly_payments']):
                sl['interest_value'+mode] = sl['dept_last'+mode]*sl['interest_rate']/sl['yearly_payments']
            elif all(string in sl for string in ['installment'+mode,'payment'+mode,'total_processing_value'+mode]):
                sl['interest_value'+mode] = sl['payment'+mode]-sl['processing_fee_value'+mode] -sl['installment'+mode]
        return sl
    def calc_total_processing_value(sl,mode=''):
        if 'total_processing_fee'+mode not in sl:
            if all(string in sl for string in ['processing_fee_value'+mode,'processing_fee_rate','dept_last'+mode,'yearly_payments']):
                sl['total_processing_value'+mode] = sl['processing_fee_value'+mode] + sl['processing_fee_rate']/sl['yearly_payments']*sl['dept_last'+mode]
            elif all(string in sl for string in ['interest_value'+mode,'payment'+mode,'installment'+mode]):
                sl['total_processing_value'+mode] = sl['payment'+mode]-sl['interest_value'+mode]-sl['installment'+mode]
        return sl
    def calc_installment(sl,mode=''):
        if 'installment'+mode not in sl:
            if all(string in sl for string in ['interest_value'+mode,'payment'+mode,'total_processing_value'+mode]):
                sl['installment'+mode] = sl['payment'+mode] - sl['total_processing_value'+mode] - sl['interest_value'+mode]
            if all(string in sl for string in ['dept'+mode,'dept_last'+mode]):
                sl['installment'+mode] = sl['dept_last'+mode] - sl['dept'+mode]
        return sl
    def calc_annuity_payment(sl):
        if 'annuity_payment' not in sl:
            if all(string in sl for string in ['total_processing_value','payment']):
                sl['annuity_payment'] = sl['payment'] - sl['total_processing_value']
        return sl
    def calc_processing_fee_rate(sl,mode=''):
        if 'processing_fee_rate' not in sl:
            if all(string in sl for string in ['processing_fee_rate_value'+mode,'dept_last'+mode,'yearly_payments']):
                sl['processing_fee_rate'] = sl['yearly_payments'] * sl['processing_fee_rate_value'+mode] / sl['dept_last'+mode]
        return sl
    def calc_processing_fee_rate_value(sl,mode=''):
        if 'processing_fee_rate_value'+mode not in sl:
            if all(string in sl for string in ['processing_fee_rate']):
                if sl['processing_fee_rate'] == 0:
                    sl['processing_fee_rate_value'+mode] = 0
                elif all(string in sl for string in ['processing_fee_rate','dept_last'+mode,'yearly_payments']):
                    sl['processing_fee_rate_value'+mode] = sl['processing_fee_rate']/ sl['yearly_payments'] / sl['dept_last'+mode]
        return sl
    def calc_dept(sl,mode=''):
        if 'dept'+mode not in sl:
            if all(string in sl for string in ['dept_last'+mode,'installment'+mode]):
                sl['dept'+mode] = sl['dept_last'+mode] - sl['installment'+mode]
            elif mode == '_last' and all(string in sl for string in ['dept','installment']):
                sl['dept'+mode] = sl['dept'] - sl['installment']
        return sl
    if calc_val == 'all' or calc_val == 'interest_rate':
        if 'interest_rate' not in sl or not isinstance(sl['interest_rate'],float):
            sl = calc_interest_rate(sl,'_last')
            sl = calc_interest_rate(sl)
    if calc_val == 'all' or calc_val == 'interest_value':
        if 'interest_value' not in sl or not isinstance(sl['interest_value'],float):
            sl = calc_interest_value(sl,'_last')
            sl = calc_interest_value(sl)
    if calc_val == 'all' or calc_val == 'processing_fee_rate':
        if 'processing_fee_rate' not in sl or not isinstance(sl['processing_fee_rate'],float):
            sl = calc_processing_fee_rate(sl,'_last')
            sl = calc_processing_fee_rate(sl)
    if calc_val == 'all' or calc_val == 'processing_fee_rate_value':
        if 'processing_fee_rate_value' not in sl or not isinstance(sl['processing_fee_rate_value'],float):
            sl = calc_processing_fee_rate_value(sl,'_last')
            sl = calc_processing_fee_rate_value(sl)
    if calc_val == 'all' or calc_val == 'total_processing_value':
        if 'total_processing_value' not in sl or not isinstance(sl['total_processing_value'],float):
            sl = calc_total_processing_value(sl,'_last')
            sl = calc_total_processing_value(sl)
    if calc_val == 'all' or calc_val == 'installment':
        if 'installment' not in sl or not isinstance(sl['installment'],float):
            sl = calc_installment(sl,'_last')
            sl = calc_installment(sl)
    if calc_val == 'all' or calc_val == 'dept':
        if 'dept_last' not in sl or not isinstance(sl['dept_last'],float):
            sl = calc_dept(sl,'_last')
        if 'dept' not in sl or not isinstance(sl['dept'],float):
            sl = calc_dept(sl)
    if calc_val == 'all' or calc_val == 'interest_value':
        if 'annuity_payment' not in sl or not isinstance(sl['annuity_payment'],float):
            sl = calc_annuity_payment(sl)
    return sl

def set_default_parameters(substituted_parameters):
    if 'processing_fee_value' not in substituted_parameters:
        substituted_parameters['processing_fee'] = 0.0
    if 'processing_fee_rate' not in substituted_parameters and 'processing_fee_rate_value' not in substituted_parameters:
        substituted_parameters['processing_fee_rate'] = 0.0
    if 'yearly_payments' not in substituted_parameters:
        substituted_parameters['yearly_payments'] = 12
    return substituted_parameters

def overview(df, mode='simple'):
    if mode == 'simple':
        # Sort dataframe by 'amount'
        df_sorted = df.sort_values(by='amount')

        # Create PrettyTable with required columns
        tab = pt(['Date', 'Text', 'Amount'])

        # Add smallest 5 rows
        df_small = df_sorted.head(5).copy()
        df_small = df_print_format(df_small)
        for _, row in df_small.iterrows():
            tab.add_row([row['date'], row['text'], row['amount']])
        for col, align in zip(tab.field_names,["c","l","r"]):
            tab.align[col] = align
        print('Smallest posts')
        print(tab)
        print()
        tab = pt(['Date', 'Text', 'Amount'])
        df_large = df_sorted.tail(5).copy()
        df_large = df_print_format(df_large)
        for _, row in df_large.iterrows():
            tab.add_row([row['date'], row['text'], row['amount']])
        for col, align in zip(tab.field_names,["c","l","r"]):
            tab.align[col] = align

        print('Largest posts')
        print(tab)
        print()
        tab = pt(['Statistic', 'Value'])
        # Calculate average amount size and sum
        tab.add_row(['Average',"{:.2f}".format(df['amount'].mean())])
        tab.add_row(['Sum',"{:.2f}".format(df['amount'].sum())])
        for col, align in zip(tab.field_names,["l","r"]):
            tab.align[col] = align
        print(tab)

        # Add summary rows
        #print(f'Average: {"{:.2f}".format(avg_amount)}')
        #print(f'Sum:     {"{:.2f}".format(total_amount)}')
        

    elif mode == 'months':
        # Calculate sum for last 12 months plus the current month
        today = datetime.today()
        end_date = today.replace(day=1)
        start_date = (end_date - timedelta(days=365)).replace(day=1)

        # Filter dataframe for the last 12 months plus the current month
        df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

        # Group by month and calculate sum of amount
        monthly_sum = df_filtered.groupby(df_filtered['date'].dt.strftime('%Y-%m'))['amount'].sum()

        # Create PrettyTable with required columns
        tab = pt(['Month', 'Sum'])

        # Add rows for each month
        for index, value in monthly_sum.items():
            tab.add_row([datetime.strptime(index, '%Y-%m').strftime('%b %Y'),"{:,.2f}".format(value)])
        print(tab)

    else:
        raise ValueError("Invalid mode. Choose either 'simple' or 'months'.")

def print_help(commands,view):
    global skip_update
    skip_update = True
    if commands == 'basic':
        print('Main commands:')
        print('i=import posts')
        print('ic=import change by id')
        print('key=to switch for predefined view and filter')
        print('u=undo latest command - r=redo latest command')
        print('l=load previous data - w=write data')
        print('q=quit')
        print('h=help, t=update, d=post detail')
        print()
        print('Switch views:')
        print('key=to switch for predefined view and filter')
        print('views=lists all predifined views in settings.views')
        print()
        print('Plot dataframe:')
        print('pls=Sankey plot of current filtered dataframe')
        print('key=to switch for predefined plots')
        print('plots=lists all predifined plots in settings.plots')
        print()
        print('Overview of filtered posts')
        print('ovs for simple overview')
        print('ovm for sum last 12 months')
        print()
        print('Change filter or values')
        print('f=filter posts - c=change column value - s=split a row')
        print('enter f,c or s for detailts on each function')
        print()
        print('Page options:')
        print('pn     Go to next page')
        print('pp     Go to previous page')
        print('p ##   Go to page number')
        print('pn ##  Go ## pages forward')
        print()
        print('Other commands')
        print('clear=Clear all loaded dataframes')
        print('del=Delete rows in dataframe.(Only recommended for entire account). Enter del for details')
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
        print('    string to filter based on a string in the column or category')
        print()
        print('For filtering on time, use either date string(2023-11-15) or relative date from today(-1year2month3day4hour5min6sec)')
        print('For filtering time on last full months, use either =l12fmon to see the last completed 12 months')
        print()
        print('Other filter options:')
        print('f reset  - reset all filter')
        print('f del column  - delete filters on a column')
        print()
        print('To filter for more than 1 option, use comma to seperate')
        print('f text apple,banana  - to filter apple and banana texts')
        print('f cat pizza,electronic  - to filter categories where the name include pizza or electronic')
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
        print()
        print('Current:')
        print(df_filters)
    elif commands == 'change':
        print('Change column options:')
        print('c ## column val')
        print()
        print(' To delete a column value, set value to ""')
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
    elif commands == 'delete':
        print('Delete rows:')
        print('del ##')
        print()
        print('Where ## can be:')
        print('    A single select number')
        print('    Multiple select numbers eg. 1,6,7')
        print('    Range of select numbers eg. 5-9')
        print('    all for all rows in all pages in current filter')
    elif commands == 'page':
        print('Change page options:')
        print('p ##')
        print()
        print('Where ## is page number:')
        print()
        print(' To go to next or previous page:')
        print('    Next: pn')
        print('    Previous: pp')
        print()
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
        print('Delete a splited id by')
        print('s del ##')
        print()

def initialize_settings():
    settings = import_settings('general')
    if len(settings) == 0:
        print('file settings.general is missing or empty,')
        exit()
    view_setting = import_settings('views')
    if len(settings) == 0:
        print('file settings.views is missing or empty, Minimum 1 view is required')
        exit()
    plot_settings = import_settings('plots')
    email_settings = import_settings('email')
    if 'filters' in view_setting:
        settings['filters'] = view_setting['filters']
    if 'views' in view_setting:
        settings['views'] = view_setting['views']
    if 'plots' in plot_settings:
        settings['plots'] = plot_settings['plots']
    settings['email'] = email_settings
    settings['marked'] = import_settings('marked')
    settings['export'] = import_settings('export')
    settings['dept'] = import_settings('dept')
    settings['category_types'],settings['categories'],settings['auto_categories'] = import_settings('categories')
    settings = convert_delta_filter_time(settings)
    return settings

def initialize_view_and_budget(settings):
    view = {}
    view["view"] = 0
    view["filter"] = 0
    view["type"] = settings['views'][0]['type']
    #default budget_view
    view["budget_view"] = ['budget','all',0] #budget/status, 1/2/3 for all/categories/types 0/1 seperate marked
    #budget_filter
    view["budget_filter"] = {}
    view["seperate_marked_in_budget"] = convert_to_bool(settings.get("seperate_marked_in_budget",True))
    view["max_col_width"] = int(settings.get("max_col_width",'32'))
    view["show_multiple_lines"] = convert_to_bool(settings.get("show_multiple_lines",True))
    view["page"] = '1'
    cur_year = datetime.now().year
    budgets = read_budget_files()
    view["year"] = 0
    if cur_year in budgets:
        view["year"] = cur_year
    elif len(budgets) > 0:
        highest_year = max(budgets, key=budgets.get)
        if highest_year > 2000: 
            view["year"] = highest_year
    return view,budgets


if __name__ == '__main__':
    # Enable history navigation
    #reorg initialize
    readline.parse_and_bind("tab: complete")
    readline.set_history_length(1000)
    settings = initialize_settings()
    view,budgets = initialize_view_and_budget(settings)
    df_filters = {}
    if 'filters' in settings:
        if 'filter' in view:
            if view['filter'] in settings['filters']:
                df_filters = settings['filters'][view["filter"]].copy()
    dfs_history = []
    current_history_index = 0
    cur_df = None
    
    print('********************** start ***********************')
    
    run = True
    skip_update = False
    dataframe_update = False
    clear_df = False
    plot_df = {}
    warn_message = []
    pickle_file_path = ''
    #reorg auto load
    if 'auto_load' in settings:
        pickle_file_path = settings['auto_load']
        if os.path.exists(pickle_file_path):
            print(f'Loading dataframe from {os.path.abspath(pickle_file_path)}')
            cur_df = pd.read_pickle(pickle_file_path)
            cur_df = check_and_correct_df(cur_df)
            view = create_output(cur_df,df_filters,view)
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
                # Set current data_frame/filter/view
                cur_df,df_filters,view = set_view_and_filter(_input,settings,view,df_filters)
                # Plot
                plot_df = set_plot(_input,settings)
                # Handle user input
                split_input = shlex.split(_input)
                cur_df,df_filters,view = handle_user_input(split_input,cur_df,df_filters,view)

                # Updates view
                if split_input[0][0] == 'q':
                    if split_input[0] == 'qw':
                        _q_input = 'w'
                    elif split_input[0] == 'q!':
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
                if plot_df:
                    plot_dataframe(cur_df, plot_df, settings)
                    plot_df = {}
                if not skip_update or dataframe_update:
                    if cur_df is not None:  
                        if view["type"] == 'posts':
                            cur_df = check_and_correct_df(cur_df)
                if not skip_update:
                    if cur_df is not None:
                        view = create_output(cur_df,df_filters,view)
                    else:
                        print('No dataframe loaded. Press l to load or i to import csv.')
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
                if clear_df:
                    dfs_history,cur_df,current_history_index = clear_dataframe()
                    clear_df = False
                    skip_update = True
                #update dfs
            except Exception as e:
                # Get the traceback information as a string
                traceback_str = traceback.format_exc()
                # Print or log the traceback string
                warn_message.append(f"Error: {traceback_str}")
                try:
                    if len(dfs_history) and view['type'] == 'posts':
                        cur_df = dfs_history[current_history_index].copy()
                        if cur_df is not None:  
                            if view["type"] == 'posts':
                                cur_df = check_and_correct_df(cur_df)
                except Exception as e:
                    traceback_str = traceback.format_exc()
                try:
                    if not skip_update or dataframe_update:
                        if cur_df is not None:  
                            if view["type"] == 'posts':
                                cur_df = check_and_correct_df(cur_df)
                except Exception as e:
                    traceback_str = traceback.format_exc()
                try:
                    if cur_df != None and len(cur_df):
                        view = create_output(cur_df,df_filters,view)
                except Exception as e:
                    traceback_str = traceback.format_exc()
                warn_message.append(f"Change command and try again.")
                warn_message.append('')
            for w in warn_message: print(w)
            warn_message = []

 
        readline.add_history(_input)
    
    
    print('********************** end ************************')
