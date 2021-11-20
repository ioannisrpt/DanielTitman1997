# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5


"""
Replicate the procedure of creating pre-formation and constant-allocation Fama-French factors as described 
in Daniel & Titman (1997).

The paper can be found at:
https://www.jstor.org/stable/2329554?seq=1#metadata_info_tab_contents


Steps to construct these new Fama-French factors
------------------------------------------------
    1. Count the number of observations (returns) for each June date (date_jun) in CRSP data
       by PERMCO or PERMNO. Augment the FirmCharacteristics.csv table with this information.
    2. Apply a rolling window of length 5 (5 years) to sum the number of observations (returns).
       Create a dummy that is 1 if the past 5-year number of observations exceeds the threshold
       and 0 otherwise. This dummy will be used to filter the set of firms (PERMCOs) or securities
       (PERMNOs) for the construction of the new Fama-French factors.
    3. Iterate through a list of June dates (list of date_jun in ascending order) and 
       apply the FFPortfolios function and get portfolio returns as in FamaFrench2015FF5 with 
       a twist:
           i. First isolate those entities that existed 5 years before.
           ii. Isolate 5-year data from the return dataframe (ret_data).
           iii. Create a new column 'date_jun_5Y' in ret_date which is just the current 
           date_jun for which we iterate through.
           iv. Rename 'date_jun' to 'date_jun_5Y' for the filtered FirmCharacterics.csv table 
           (firmchars).
           v. Apply FFPortfolios as follows for the HML factor:
               
sizebtm = FFPortfolios(ret_data, firmchars, entity_id = 'PERMCO', time_id  = 'date_jun_5Y', \
                       ret_time_id = 'date', characteristics = ['CAP', 'BtM'], lagged_periods = [0, 0], \
                       [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                       ffdir = FFDIR, conditional_sort =  False, weight_col = 'CAP')
               
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
# BtM : 1 = Low, 2 = Neutral, 3 = High
sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
               '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
    
# Isolate the portfolios and rename the columns
# Also drop the first year as defined in the first index element of 'num_stocks'
sizebtm_p = sizebtm['ports'][sizebtm['ports'].index > sizebtm['num_stocks'].index[0]].copy().rename(columns = sizebtm_def)

# Define the HML factor
sizebtm_p['HML_DT97'] = (1/2)*(sizebtm_p['SH'] + sizebtm_p['BH']) - \
                        (1/2)*(sizebtm_p['SL'] + sizebtm_p['BL']) 
                        
          vi. Save the HML_DT97 in a dataframe that has three columns; 
          'aate' = monthly or daily returns of the factors 
          'date_jun_5Y' = current date_jun being iterated
          'HML_DT97' = pre-formation and constant-weight allocation HML factor.
          vii. Concat all HML_DT97 dataframes on axis = 0. 
          
               
"""



import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BYearEnd
from pandas.tseries.offsets import BMonthEnd
from functools import reduce
import matplotlib.pyplot as plt
# Python time counter 
from time import perf_counter


# Main directory
wdir = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\DanielTitman1997'
os.chdir(wdir)
# Portfolio directory 
FFDIR = r'C:\Users\ropot\OneDrive\Desktop\Python Scripts\DanielTitman1997\FFfactorsDT97'


# Import the PortSort Class. For more details: 
# https://github.com/ioannisrpt/PortSort.git 
from PortSort import PortSort


# ------------------
# Control execution
# ------------------

# DT 97 SMB factor
do_SMB = False
# DT 97 HML factor
do_HML = False
# DT 97 RMW factor
do_RMW = False
# DT 97 CMA factor
do_CMA = False
# DT 97 MKT factor
do_MKT = True
# Get Figure 1 from Daniel & Titman (1997) paper
do_figure1 = True


# -------------------------------------------------------------------------------------
#                FUNCTIONS - START
# ------------------------------------------------------------------------------------


def WeightedMean(x, df, weights):
    """
    Define the weighted mean function
    """
    return np.average(x, weights = df.loc[x.index, weights])

def is_number(s):
    """
    Function that checks if a string is a number    
    """
    # Try converting the string to float number. 
    try:
        float(s)
        return True
    # If the conversion fails, a ValueError is raised so
    # we know that the string is not a float number.
    except ValueError:
        return False
    
    
    
def ForceDatetime(x, force_value, date_format = '%Y%m%d'):
    """
    Function that converts a variable to a datetime object. If the conversion is not 
    possible, force_value is applied.

    """
    try:
        return pd.to_datetime(x, format = date_format)
    except ValueError:
        return force_value
    
    
    
def JuneScheme(x, num_format = False):
    """
    Use the June-June scheme as in Fama-French.
    
    x must be a datetime object
    """
    # Get month and year
    month = x.month
    year = x.year
    if month <= 6:
        # New date in string format 
        june_dt = '%s-06-01' % year
        y = pd.to_datetime(june_dt, format = '%Y-%m-%d')
        if num_format:
            return BMonthEnd().rollforward(y).strftime('%Y%m%d')
        else:
            return BMonthEnd().rollforward(y)
    else:
        nyear = year + 1
        # New date in string format 
        june_dt = '%s-06-01' % nyear
        y = pd.to_datetime(june_dt, format = '%Y-%m-%d')
        if num_format:
            return BMonthEnd().rollforward(y).strftime('%Y%m%d')
        else:
            return BMonthEnd().rollforward(y)
        
# Function that constructs the portfolios per Fama-French methodology 
# along with their market cap weighted returns.
def FFPortfolios(ret_data, firmchars, entity_id, time_id, ret_time_id, characteristics, lagged_periods, \
                 N_portfolios, quantile_filters, ffdir, conditional_sort = True, weight_col = 'CAP_W', \
                 return_col = 'RET'):
    """
    
    Parameters
    ----------
    ret_data : Dataframe
        Dataframe where returns for entities are stored in a panel format.
    firmchars : Dataframe
        The characteristics of the entities in ret_data on date_jun.
    entity_id : str
        Entity identifier as found in both ret_data and firmchars.
    time_id : str
        Time identifier as found in both ret_data and firmchars.
    ret_time_id : str
        Time identifier as found in ret_data. ret_time_id dicates the frequency for which market cap
        returns of the portfolios are calculated.
    characteristics : list
        A list of up to three characteristics for which entities will be sorted.
    lagged_periods : list
        A list of the number of lagged periods for the characteristics to be sorted.
        The length of characteristics and lagged_periods must match.
    N_portfolios : list 
        N_portfolios is a list of n_portfolios.
        If n_portfolios then stocks will be sorted in N_portfolios equal portfolios.
        If n_portfolios is an array then this array represents the quantiles. 
    quantile_filters : list
        It is a list of lists. Each element corresponds to filtering entities for the 
        ranking of portfolios into each firm characteristic. The lenght of the list must 
        be the same as that of firm_characteristics.
    ffdir : directory
        Saving directory.
    conditional_sort : boolean, Default=True
        If True, all sorts are conditional. If False, all sorts are unonconditional and 
        independent of each other. 
    weight_col : str, Default='CAP_W'
        The column used for weighting the returns in a portfolio. Default is 'CAP_W' which
        corresponds to the market capitalization of the previous period as defined by 'time_id'.
    return_col : str, Default='RET'
        The column of ret_data that corresponds to returns. Default value is 'RET' which is the 
        name of return for CRSP data.

    Returns
    -------
    port_dict : dictionary
        Directory with items:
            'ports' = portfolio returns
            'num_stocks' = number of stocks in each portfolio
            

    """
    
    # Drop observations if CAP_W is null
    firmchars = firmchars.dropna(subset = [weight_col])
    
    # Define the class using the first sorting characteristic
    port_char = PortSort(firmchars, firm_characteristic = characteristics[0], \
                         lagged_periods = lagged_periods[0], n_portfolios = N_portfolios[0], \
                         entity_id = entity_id, time_id = time_id, quantile_filter = quantile_filters[0], \
                         save_dir = ffdir)
    
    # -----------------------------------
    #  SORT -- SINGLE or DOUBLE or TRIPLE
    # -----------------------------------
    
    # One characteristic --> Single Sort
    # ----------------------------------
    if len(characteristics) == 1:
        
        # Univariate sort
        port_char.SingleSort()  
        
        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.portfolio
        ports = port_char.single_sorted[[time_id, entity_id, weight_col, port_name]].copy()    
        
        
        # Define save names
        save_str =  '%d_portfolios_sortedBy_%s.csv' % (port_char.num_portfolios, characteristics[0])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
        
    # Two characteristic --> Double Conditional Sort
    # -----------------------------------------------
    if len(characteristics) == 2:
                              
        # Bivariate sort
        port_char.DoubleSort(characteristics[1], lagged_periods[1], N_portfolios[1], quantile_filter_2 = quantile_filters[1], \
                             conditional = conditional_sort, save_DoubleSort = False)   
        
        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.double_sorted.columns[-1]
        ports = port_char.double_sorted[[time_id, entity_id, weight_col, port_name]].copy()
        
        # Define save names
        save_str =  '%dx%d_portfolios_sortedBy_%s_and_%s.csv' % ( port_char.num_portfolios, \
                                                                    port_char.num_portfolios_2, \
                                                                    characteristics[0], characteristics[1])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
        
    
    # Three characteristics --> Triple Conditional Sort
    # -------------------------------------------------
    if len(characteristics) == 3:
        
        # Triple sort
        port_char.TripleSort(characteristics[1], characteristics[2], lagged_periods[1], lagged_periods[2], \
                             n_portfolios_2  = N_portfolios[1], n_portfolios_3 = N_portfolios[2], \
                             quantile_filter_2 = quantile_filters[1], quantile_filter_3 = quantile_filters[2], \
                             conditional = conditional_sort, save_TripleSort = False)

        # Isolate only the essential columns for portfolio assignment
        port_name = port_char.triple_sorted.columns[-1]
        ports = port_char.triple_sorted[[time_id, entity_id, weight_col, port_name]].copy()       

        
        # Define save names
        save_str =  '%dx%dx%d_portfolios_sortedBy_%s_and_%s_and_%s.csv' % ( port_char.num_portfolios, \
                                                                    port_char.num_portfolios_2, \
                                                                    port_char.num_portfolios_3,
                                                                    characteristics[0], characteristics[1],\
                                                                    characteristics[2])
        save_ret = 'RET_' + save_str
        save_num = 'NUM_STOCKS_' + save_str
            

    
    
    # Number of stocks in a portfolio
    # -------------------------------
    num_stocks = ports.groupby(by = [port_name, port_char.time_id] )[port_name].count().unstack(level=0)
    
    
    # --------------------------------------------------
    #  ASSIGN PORTFOLIOS TO RETURN DATA (MONTHLY OR DAILY)
    # --------------------------------------------------
    
    # The inner merging is taking care of stocks that should be excluded from the formation of the portfolios
    ret_ports = pd.merge(ret_data, ports, how = 'inner', on = [time_id, entity_id], suffixes = ('', '_2') )
    
    char_ports = ret_ports.groupby(by = [port_name, ret_time_id] ).agg( { return_col : lambda x: WeightedMean(x, df = ret_ports, weights = weight_col) } ).unstack(level=0)
    # Rename the columns by keeping only the second element of their names
    char_ports.columns = [x[1] for x in char_ports.columns]
    
    #-------------
    # SAVE RESULTS
    # ------------
            
    char_ports.to_csv(os.path.join(ffdir, save_ret ))
    num_stocks.to_csv(os.path.join(ffdir, save_num ))
    
    # Put everyting in a dictionary
    port_dict = {}
    port_dict['ports'] = char_ports
    port_dict['num_stocks'] = num_stocks
    

    return port_dict


# Compounded return from returns with a threshold for not null returns
def compReturn(returns, threshold = 0):
    # Maybe change in "if not returns:" ?
    if not returns.empty:
        if pd.notnull(returns).sum() >= threshold:
            return ((returns + 1).prod()) - 1
        else:
            return np.nan
    else:
        return np.nan
    
# -------------------------------------------------------------------------------------
#                FUNCTIONS - END
# ------------------------------------------------------------------------------------ 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                      IMPORT-FORMAT DATA                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ---------
# CRSP data
# ---------

# Import CRSP monthly data used for extracting the returns of the Fama-French factors
crsp = pd.read_csv(os.path.join(wdir, 'CRSPmonthlydata1963.csv'))
# Keep only certain columns
crspm = crsp[['PERMCO', 'date', 'RET', 'date_jun']].copy()
del crsp

# Show the format of crspm
print(crspm.head(15))

# --------------------
# FIRM CHARACTERISTICS
# --------------------

# Import FirmCharacteristics table used for sorting stocks in portfolios as created from
# FamaFrench2015FF5.
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristics2.csv'))
# Sort values to make sure that sorting and everything else will be working as intended 
firmchars = firmchars.sort_values(by = ['PERMCO', 'date_jun'])


# --------------------
#  FAMA-FRENCH FACTORS
# --------------------

# Import the five Fama-French factors 
ff5 = pd.read_csv(os.path.join(wdir, 'FF5_monthly.csv')).dropna().astype({'date' : np.int64})



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            CALCULATE NUMBER OF RETURN OBSERVATIONS                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Get number of observations from CRSP data
num_ret = crspm.groupby(['PERMCO', 'date_jun'])['RET'].count().reset_index().rename(columns = {'RET' : 'NUM_RET'})
# Merge with firmchars
firmchars = pd.merge(firmchars, num_ret, on = ['PERMCO', 'date_jun'])
# Calculate the 5-Year number of valid return observations
firmchars['NUM_RET_5Y'] = firmchars.groupby('PERMCO')['NUM_RET'].transform(lambda s: s.rolling(5).sum())
# Create a dummy to check if NUM_RET_5Y exceeds threshold
firmchars['exists_5Y'] = firmchars['NUM_RET_5Y'].apply(lambda x: 1 if x >= 58 else 0)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             ISOLATE THE FORMATION AND HOLDING DATES                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Isolate the formation dates which coincide with date_jun
fdates = pd.DataFrame(data = firmchars['date_jun'].drop_duplicates().sort_values().reset_index(drop = True), columns = ['date_jun'])
# Get the holding date 5 years before
fdates['date_hold'] = fdates['date_jun'].shift(4)
# Isolate only the valid formation and holding dates
fdates = fdates.dropna().reset_index(drop = True).astype({'date_hold' : np.int64})
# Drop the last formation date which corresponds to 20210630 
fdates = fdates.iloc[:-1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                DANIEL & TITMAN (1997) METHODOLOGY                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Save the DT97 factors in dataframes
SMB_DT97 = pd.DataFrame()
HML_DT97 = pd.DataFrame()
RMW_DT97 = pd.DataFrame()
CMA_DT97 = pd.DataFrame()
MKT_DT97 = pd.DataFrame()


# Iterate through the formation dates
for formation_date in fdates['date_jun']:
    
    print('---- %d -----' % formation_date)
    # Get the holding date
    hold_date = fdates.loc[fdates['date_jun'] == formation_date, 'date_hold'].values[0]
    
    # Isolate the firmchars of the entities that existed 5 years before given the formation date
    firmchars5Y = firmchars[ (firmchars['exists_5Y'] == 1) & (firmchars['date_jun'] == formation_date )].copy()
    # Isolate the entities themselves
    entities5Y = list(firmchars5Y['PERMCO'].drop_duplicates().values)
    
    # Isolate the return data of the same entities
    crspm5Y = crspm[ (crspm['PERMCO'].isin(entities5Y) ) &  (hold_date <= crspm['date_jun']) & ( crspm['date_jun']<= formation_date ) ].copy()
    
    # Replace the values of the 'date_jun' column with formation_date so that FFPortfolios will work
    # as intended in the context of the construction of the new Fama-French factors
    crspm5Y['date_jun'] = formation_date
    

    
    # ~~~~~~~~~~~~~
    # SMB FACTOR  #
    # ~~~~~~~~~~~~~ 
    
    # Control execution
    # -----------------
    if do_SMB:
    
        print('SMB factor')
        
        # Apply the FFPortfolios to create the 2 Size portfolios 
        size = FFPortfolios(crspm5Y, firmchars5Y, entity_id = 'PERMCO', time_id  = 'date_jun', \
                           ret_time_id = 'date', characteristics = ['CAP'], lagged_periods = [0],
                           N_portfolios = [2], quantile_filters = [['EXCHCD', 1]], \
                           ffdir = FFDIR, conditional_sort = False, weight_col = 'CAP')
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        size_def = {1 : 'S', 2 : 'B'}
        
        
            
        # Isolate the portfolios and rename the columns
        size_p = size['ports'].copy().rename(columns = size_def)
        
        # Define the SMB factor (simplest form)
        size_p['SMB_DT97']  = size_p['S'] - size_p['B']
        
        # Put the new SMB factor in a DataFrame
        smb_DT97 = pd.DataFrame(data = size_p['SMB_DT97'].reset_index())
        smb_DT97['date_jun'] = formation_date
        # and concat    
        SMB_DT97 = pd.concat([SMB_DT97, smb_DT97], axis = 0)    
        
    # ~~~~~~~~~~~~~
    # HML FACTOR  #
    # ~~~~~~~~~~~~~
    
    # Control execution
    # -----------------
    if do_HML:
    
        print('HML factor')
        
        # Apply the FFPortfolios to create the 2x3 Size and Book-to-Market portfolios 
        sizebtm = FFPortfolios(crspm5Y, firmchars5Y, entity_id = 'PERMCO', time_id  = 'date_jun', \
                           ret_time_id = 'date', characteristics = ['CAP', 'BtM'], lagged_periods = [0, 0], \
                           N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                           ffdir = FFDIR, conditional_sort =  False, weight_col = 'CAP')
                   
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # BtM : 1 = Low, 2 = Neutral, 3 = High
        sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
                       '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
        
        # Isolate the portfolios and rename the columns
        sizebtm_p = sizebtm['ports'].copy().rename(columns = sizebtm_def)
        
        # Define the HML factor
        sizebtm_p['HML_DT97'] = (1/2)*(sizebtm_p['SH'] + sizebtm_p['BH']) - \
                                (1/2)*(sizebtm_p['SL'] + sizebtm_p['BL']) 
                                
                                
        # Put the new HML factor in a DataFrame
        hml_DT97 = pd.DataFrame(data = sizebtm_p['HML_DT97'].reset_index())
        hml_DT97['date_jun'] = formation_date
        # and concat
        HML_DT97 = pd.concat([HML_DT97, hml_DT97], axis = 0)
    
    
    # ~~~~~~~~~~~~~
    # RMW FACTOR  #
    # ~~~~~~~~~~~~~
    
    # Control execution
    # -----------------
    if do_RMW:

        print('RMW factor')
        
        # Create the 2x3 Size and Profitability portfolios 
        sizermw = FFPortfolios(crspm5Y, firmchars5Y, entity_id = 'PERMCO', time_id  = 'date_jun', \
                           ret_time_id = 'date', characteristics = ['CAP', 'OP'], lagged_periods = [0, 0], \
                           N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                           ffdir = FFDIR, conditional_sort = False, weight_col = 'CAP')
    
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # OP : 1 = Weak, 2 = Neutral, 3 = Robust
        sizermw_def = {'1_1' : 'SW', '1_2' : 'SN', '1_3' : 'SR', \
                       '2_1' : 'BW', '2_2' : 'BN', '2_3' : 'BR'}
        
        # Isolate the portfolios and rename the columns
        sizermw_p = sizermw['ports'].copy().rename(columns = sizermw_def)
    
        # Define the RMW factor
        sizermw_p['RMW_DT97'] = (1/2)*(sizermw_p['SR'] + sizermw_p['BR']) - \
                             (1/2)*(sizermw_p['SW'] + sizermw_p['BW'])   
                             
                             
        # Put the new HML factor in a DataFrame
        rmw_DT97 = pd.DataFrame(data = sizermw_p['RMW_DT97'].reset_index())
        rmw_DT97['date_jun'] = formation_date
        # and concat
        RMW_DT97 = pd.concat([RMW_DT97, rmw_DT97], axis = 0)
    
    # ~~~~~~~~~~~~~
    # CMA FACTOR  #
    # ~~~~~~~~~~~~~
    
    # Control execution
    # -----------------
    if do_CMA:

        print('CMA factor')   
     
        # Create the 2x3 Size and Investment portfolios 
        sizecma = FFPortfolios(crspm5Y, firmchars5Y, entity_id = 'PERMCO', time_id = 'date_jun', \
                               ret_time_id = 'date', characteristics = ['CAP', 'INV'], lagged_periods = [0,0], \
                               N_portfolios = [2, np.array([0, 0.3, 0.7]) ], quantile_filters = [['EXCHCD', 1], ['EXCHCD', 1]], \
                               ffdir = FFDIR, conditional_sort = False, weight_col = 'CAP')
    
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # INV : 1 = Conservative, 2 = Neutral, 3 = Aggressive
        sizecma_def = {'1_1' : 'SC', '1_2' : 'SN', '1_3' : 'SA', \
                       '2_1' : 'BC', '2_2' : 'BN', '2_3' : 'BA'}
        
        # Isolate the portfolios and rename the columns
        sizecma_p = sizecma['ports'].copy().rename(columns = sizecma_def)
    
        # Define the CMA factor
        sizecma_p['CMA_DT97'] = (1/2)*(sizecma_p['SC'] + sizecma_p['BC']) - \
                             (1/2)*(sizecma_p['SA'] + sizecma_p['BA'])      
    
        
        # Put the new HML factor in a DataFrame
        cma_DT97 = pd.DataFrame(data = sizecma_p['CMA_DT97'].reset_index())
        cma_DT97['date_jun'] = formation_date
        # and concat
        CMA_DT97 = pd.concat([CMA_DT97, cma_DT97], axis = 0)
        
        
    # ~~~~~~~~~~~~~
    # MKT FACTOR  #
    # ~~~~~~~~~~~~~ 
    
    # Control execution
    # -----------------
    if do_MKT:
    
        print('MKT factor')
        
        # Get the market capitalization (CAP) of all stocks existing for 5 years 
        crspmkt = pd.merge(crspm5Y, firmchars5Y[['date_jun', 'PERMCO', 'CAP']], on = ['date_jun', 'PERMCO']).dropna()
        
        # Calculate the market return 
        mkt_DT97 = crspmkt.groupby(by = 'date').agg( {'RET' : lambda x: WeightedMean(x, df=crspmkt, weights = 'CAP') } )
        # Rename the column
        mkt_DT97.columns = ['MKT_DT97']    
        # Get the excess return of the market portfolio
        mkt_DT97 = mkt_DT97.join(ff5.set_index('date')['RF'])
        mkt_DT97['MKT_DT97'] = mkt_DT97['MKT_DT97']  - mkt_DT97['RF']
        
        # Restructure the DataFrame
        mkt_DT97 = mkt_DT97['MKT_DT97'].reset_index()
        mkt_DT97['date_jun'] = formation_date
        # and concat    
        MKT_DT97 = pd.concat([MKT_DT97, mkt_DT97], axis = 0)    
        
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               PUTTING EVERYTHING TOGETHER AND SAVE                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# List to save all 
l = []
    
if do_SMB:
    SMB_DT97 =  SMB_DT97.reset_index(drop = True)
    SMB_DT97.to_csv(os.path.join(wdir, 'SMB_DT97.csv'), index = False)
    print('SMB_DT_97 is saved.')
    l.append(SMB_DT97)

if do_HML:
    HML_DT97 =  HML_DT97.reset_index(drop = True)
    HML_DT97.to_csv(os.path.join(wdir, 'HML_DT97.csv'), index = False)
    print('HML_DT_97 is saved.')
    l.append(HML_DT97)
    
if do_RMW:
    RMW_DT97 =  RMW_DT97.reset_index(drop = True)
    RMW_DT97.to_csv(os.path.join(wdir, 'RMW_DT97.csv'), index = False)
    print('RMW_DT_97 is saved.')
    l.append(RMW_DT97)

if do_CMA:
    CMA_DT97 =  CMA_DT97.reset_index(drop = True)
    CMA_DT97.to_csv(os.path.join(wdir, 'CMA_DT97.csv'), index = False)
    print('CMA_DT_97 is saved.')
    l.append(CMA_DT97)
    
if do_MKT:
    MKT_DT97 =  MKT_DT97.reset_index(drop = True)
    MKT_DT97.to_csv(os.path.join(wdir, 'MKT_DT97.csv'), index = False)
    print('MKT_DT_97 is saved.') 
    l.append(MKT_DT97)


FF5_DT97 = reduce(lambda a, b : pd.merge(a,b, how = 'inner', on = ['date', 'date_jun']), l)
FF5_DT97.to_csv(os.path.join(wdir, 'FF5_DT97.csv'), index = False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              FIGURE 1 IN DANIEL & TITMAN (1997) PAPER                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Control execution
if do_figure1:
    
    # Import the FF5_DT97
    FF5_DT97 = pd.read_csv(os.path.join(wdir, 'FF5_DT97.csv'))
    
    
    # Function that returns the number of months between 2 dates
    def num_months(startdt, enddt):
        startdt = pd.to_datetime(startdt, format = '%Y%m%d')
        enddt = pd.to_datetime(enddt, format = '%Y%m%d')
        return (enddt.year - startdt.year)*12 + enddt.month - startdt.month   
    
    # Define the number of months before the formation date
    FF5_DT97['Months_before'] = - FF5_DT97.apply(lambda x: num_months(x['date'], x['date_jun']), axis = 1)

    # Get the average monthly return for the DT97 HML factor
    hml_ret = FF5_DT97.groupby('Months_before')['HML_DT97'].mean()
    
    # Plot 
    plt.figure()
    hml_ret.plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97.png'))
    plt.close()