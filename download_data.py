import pandas as pd
import numpy as np

def download_stock_data(symbol, start, end):
    """Downloads historical stock pricing data for an individual symbol"""

    df = pd.DataFrame()  

    # Print the symbol being downloaded
    print('DOWNLOADING ' + symbol, sep=',', end=' ... ', flush=True)  

    try:
        # Download pricing data
        stock = []
        stock = yf.download(symbol, start=start, end=end, progress=False)

        # Append individual prices
        if len(stock) == 0:
            None
        else:
            stock['Name'] = symbol
            df = df.append(stock,sort=False)
            df = df[['Close', 'Name']]
            print('SUCCESS!')
            
    except Exception:
        print('ERROR IN DOWNLOADING DATA')
               
    return df