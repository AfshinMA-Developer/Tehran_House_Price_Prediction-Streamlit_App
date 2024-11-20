import re, os
from bs4 import BeautifulSoup
import pandas as pd
import requests 

def convert_to_persian_numbers(text: str) -> str:
    """Convert English numbers in a string to Persian numbers."""
    english_to_persian = {
        '0': '۰',
        '1': '۱',
        '2': '۲',
        '3': '۳',
        '4': '۴',
        '5': '۵',
        '6': '۶',
        '7': '۷',
        '8': '۸',
        '9': '۹'
    }

    # Replace English digits with Persian digits
    for eng_digit, persian_digit in english_to_persian.items():
        text = text.replace(eng_digit, persian_digit)

    return text

def get_USD_to_IR() -> float:
    url = 'https://www.tgju.org/profile/price_dollar_rl'
    response = requests.get(url)
    if response.status_code != 200:
        print(f'> Error in fetching {url}: {response.status_code}.')
        return 300000
    soup = BeautifulSoup(response.text, 'html.parser')
    per_usd = soup.find('span', {'data-col' : 'info.last_trade.PDrCotVal'}).text.replace(',', '')
    return float(convert_to_persian_numbers(per_usd))

def remove_outliers_iqr(data:pd.DataFrame, column_name:str, threshold:float= 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    return data[~((data[column_name] < (Q1 - threshold * IQR)) | (data[column_name] > (Q3 + threshold * IQR)))]

def data_cleaning(path:str) -> pd.DataFrame:
    # Step 1 : Load and Prepare data
    # I : Load the dataset
    cleaned_data_path = path.rsplit('/', 1)[0] + '/cleaned_housePrice.csv' 
    if os.path.exists(cleaned_data_path): 
        return pd.read_csv(cleaned_data_path)
    else: 
        df = pd.read_csv(path)

    # II : Update the **Price**
    today_usd = get_USD_to_IR()
    # Every USD is equal to 30,000 Tomans (Extra Info).
    correct_coeff = today_usd / 300000
    df.Price = df.Price.apply(lambda x: x * correct_coeff * 10)

    # III : Drop irrequired columns
    df = df.drop(['Price(USD)'], axis= 1)

    # Step 2 : Data Cleaning
    # I : Correct the datatype of columns
    df.Area = df.Area.apply(lambda x: re.sub(r'\D', '', str(x)))
    df.Area = pd.to_numeric(df.Area, errors= 'coerce')

    # II : Handle **missing values**
    # Drop Null Values
    df.dropna(ignore_index= True, inplace= True)

    # III : Handle **duplicates**
    df = df.drop_duplicates(ignore_index= True)

    # IV : Handle **outliers**
    df = remove_outliers_iqr(df, 'Price')
    df = remove_outliers_iqr(df, 'Area')
    df.reset_index(drop= True, inplace= True)

    # V : Save the cleaned dataset
    df.to_csv(path.rsplit('/', 1)[0] + 'cleaned_housePrice.csv', index= False)

    return df

