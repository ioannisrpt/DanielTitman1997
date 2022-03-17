# -*- coding: utf-8 -*-
# Python 3.7.7
# Pandas 1.0.5
# Author: Ioannis Ropotos

"""
Replicate the procedure of creating pre-formation and constant-allocation Fama-French factors as described 
in Daniel & Titman (1997).

The paper can be found at:
https://www.jstor.org/stable/2329554?seq=1#metadata_info_tab_contents


Logical steps behind the construction of new Fama-French factors
-----------------------------------------------------------------
    1. Count the number of observations (returns) for each June date (date_jun) in CRSP data
       by PERMCO or PERMNO. Augment the FirmCharacteristics.csv table with this information.
       If a date lies between the start of July of year t and the end of June of year t+1, it is 
       mapped to the end of June of year t+1. Column 'date_jun' contains exactly this mapping. 
       For example date = 20200801 is mapped to 20210631 and date = 20200515 to 20200631. 
    2. Apply a rolling window of length 5 (5 years) to sum the number of observations (returns).
       Create a dummy that is 1 if the past 5-year number of observations exceeds the threshold
       and 0 otherwise. This dummy will be used to filter the set of firms (PERMCOs) or securities
       (PERMNOs) for the construction of the new Fama-French factors.
    3. Iterate through a list of June dates (list of date_jun in ascending order) and 
       apply the FFPortfolios() method and get portfolio returns as in FamaFrench2015FF5 with 
       a twist:
           i. First isolate those entities that existed 5 years before.
           ii. Isolate 5-year data from the return dataframe (ret_data).
           iii. Set 'date_jun' as the formation date of the portfolio for the 5-Year filtered
           return dataset. 
           iv. The FirmCharacteristics table has already the formation date as 'date_jun' from
           the filtering procedure.
           v. Apply the FFFPortfolios() method. 
           
           I present the process for the HML factor:
               
sizebtm = FFPortfolios(ret_data, firmchars, 
                       ret_time_id = 'date_m', 
                       FFcharacteristics = ['ME', 'BtM'], 
                       FFlagged_periods = [0, 0], 
                       FFn_portfolios = [2, np.array([0, 0.3, 0.7]) ], 
                       FFquantile_filters = [['NYSE', 1], ['NYSE', 1]], 
                       FFdir = FFDIR, 
                       FFconditional = [False],
                       weight_col = 'CAP')
               
# Renaming the portfolios as per Fama & French (2015) 
# Size : 1 = Small, 2 = Big
# BtM : 1 = Low, 2 = Neutral, 3 = High
sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
               '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
    
# Isolate the portfolios and rename the columns
sizebtm_p = sizebtm.FFportfolios.copy().rename(columns = sizebtm_def)

# Define the HML factor
sizebtm_p['HML_DT97'] = (1/2)*(sizebtm_p['SH'] + sizebtm_p['BH']) - \
                        (1/2)*(sizebtm_p['SL'] + sizebtm_p['BL']) 
                        
          vi. Save the HML_DT97 in a dataframe that has three columns; 
          'aate_m' for monthly or 'date' for daily returns of the factors 
          'date_jun' = formation date 
          'HML_DT97' = pre-formation and constant-weight allocation HML factor.
          vii. Concat all HML_DT97 dataframes on axis = 0. 

          


Inputs:
-------
    1. CRSPreturn1926m.csv : 
        Return CRSP dataframe with the columns
            PERMNO = security identifier
            RET = total return for the currect period
            date_m = month-year in the integer format YYYYmm
            date_jun = end of July-June period in the integer format YYYYmm
        All CRSP stocks are included in CRSPreturn1926m.csv.
    2. FF3_monthly.csv : 
        Three Fama-French factors as downloaded from Fama-French library.
        I need this file to extract the risk-free rate.
    3. FirmCharacteristicsFF5_last_traded.csv :
        Dataframe that contains the firm characteristics of all firms in 
        Compustat necessary to construct the Fama-French factors. 
        
    
Returns:
--------
    SMB_DT97_monthy.csv
    HML_DT97_monthy.csv
    CMA_DT97_monthy.csv    
    RMW_DT97_monthy.csv   
    MKT_DT97_monthy.csv
    FF5_DT97_monthy.csv 
    
    
    Figure_1_DanielTitman97_HML.png    
    Figure_1_DanielTitman97_SMB.png
    Figure_1_DanielTitman97_RMW.png
    Figure_1_DanielTitman97_CMA.png
    Figure_1_DanielTitman97_MKT.png
               
"""



import os
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
# Import the PortSort Class. For more details: 
# https://github.com/ioannisrpt/PortSort.git 
from PortSort import PortSort as ps



# Main directory (Change it)
wdir = r'C:\Users\ropot\Desktop\Python Scripts\DanielTitman1997'
os.chdir(wdir)
# Fama-French portfolio directory 
ff_folder = 'FF5_portfolios'
ff_dir = os.path.join(wdir, ff_folder)
if ff_folder not in os.listdir(wdir):
    os.mkdir(ff_dir)



# ------------------
# Control execution
# ------------------

# DT 97 SMB factor
do_SMB = True
# DT 97 HML factor
do_HML = True
# DT 97 RMW factor
do_RMW = True
# DT 97 CMA factor
do_CMA = True
# DT 97 MKT factor
do_MKT = True
# Get Figure 1 from Daniel & Titman (1997) paper
do_figure1 = True


# -------------------------------------------------------------------------------------
#                FUNCTIONS - START
# ------------------------------------------------------------------------------------


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# WEIGHTED MEAN IN A DATAFRAME    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Weighted mean ignoring nan values 
def WeightedMean(x, df, weights):
    """
    Define the weighted mean function
    """
    # Mask both the values and the associated weights
    ma_x = np.ma.MaskedArray(x, mask = np.isnan(x))
    w = df.loc[x.index, weights]
    ma_w = np.ma.MaskedArray(w, mask = np.isnan(w))
    return np.average(ma_x, weights = ma_w)



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
    
  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   MAP DATES TO JUNE DATES       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
        
def JuneScheme(x):
    """
    Use the June-June scheme as in Fama-French.
    
    x must be a datetime object. It returns a June date
    in the integer format of YYYYmm. 
    """
    # Get month and year
    month = x.month
    year = x.year

    # x is mapped to a June date
    if month<=6:
        date_jun = year*100 + 6
    else:
        nyear = year + 1
        date_jun = nyear*100 + 6
            
    return date_jun
            
    
# Function that inputs a dataframe and a date column that applies the June Scheme
# thus creating a new column named 'date_jun'
def ApplyJuneScheme(df, date_col = 'date', date_format = '%Y%m%d'):
    # Isolate the dates in date_col in a separate dataframe    
    dates = pd.DataFrame(df[date_col].drop_duplicates().sort_values(), columns = [date_col])
    # Define the June date column
    dates['date_jun'] = pd.to_datetime(dates[date_col], format = date_format).apply(lambda x: JuneScheme(x)).astype(np.int32)
    # Merge with original dataframe df. 
    # The above process is very efficient since we don't have to deal
    # with all rows of df but only with one set of dates.
    df = pd.merge(df, dates, how = 'left', on = [date_col])
    return df
   

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   COMPOUNDING RETURNS      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


print('Import data - Start \n')

# ---------
# CRSP data
# ---------


# Import CRSP data
ctotype32 = {'date_m' : np.int32,
             'date_jun' : np.int32,
             'PERMNO' : np.int32,
             'RET' : np.float32}

crspm = pd.read_csv(os.path.join(wdir, 'CRSPreturn1926m.csv')).astype(ctotype32).dropna()

# Since our sample of Book equity characteristics are 
# readily available from 196306 and onwards and we 
# we need 5 years of data before 196306, we subset
# CRSP data from 19580601; 19580601 - 19630630 is 
# 5 years of daily returns.
crspm = crspm[crspm['date_m']>=195807].copy()



# Show the format of crsp data
print(crspm.head(15))


# --------------------
# FIRM CHARACTERISTICS
# --------------------

# Import FirmCharacteristics table


ftotype32 = {'GVKEY' : np.float32,
             'PERMNO' : np.int32,
             'EXCHCD' : np.float32,
             'SHRCD' : np.float32,
             'date_jun' : np.int32,
             'ceq' : np.float32,
             'be' : np.float32,
             'operpro' : np.float32,
             'OP' : np.float32,
             'INV' : np.float32,
             'CAP' : np.float32,
             'CAP_dec' : np.float32,
             'CAP_W' : np.float32,
             'ME' : np.float32,
             'ME_dec' : np.float32,
             'ME_W' : np.float32,
             'BtM' : np.float32}
             




#firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristics2.csv')).astype(ftotype32)
firmchars = pd.read_csv(os.path.join(wdir, 'FirmCharacteristicsFF5_last_traded.csv')).astype(ftotype32)
# Drop any other column that is not in ftotype32 keys
drop_fcols = list( set(firmchars.columns) - set(list(ftotype32.keys())) )
firmchars.drop(columns = drop_fcols, inplace = True)
# Subset for EXCHCD (NYSE, AMEX and NASDAQ)
firmchars = firmchars.dropna(subset = ['EXCHCD'])
firmchars = firmchars[firmchars['EXCHCD'].isin(set([1,2,3]))]
# Define NYSE stocks for constructing breakpoints
nyse1 = firmchars['EXCHCD'] == 1
nyse2 = firmchars['SHRCD'] == 10.0
nyse3 = firmchars['SHRCD'] == 11.0
firmchars['NYSE'] = np.where(nyse1 & ( nyse2 | nyse3), 1, 0)
# Subset for ordinary common shares
shrcd = [10, 11]
firmchars = firmchars[firmchars['SHRCD'].isin(set(shrcd))].copy()
# Subset for time
firmchars = firmchars[firmchars['date_jun']>=196306]
# Sort values to make sure that sorting and everything else will be working as intended 
firmchars = firmchars.sort_values(by = ['PERMNO', 'date_jun'])


# --------------------
#  FAMA-FRENCH FACTORS
# --------------------

# Import the 3 Fama-French factors (monthly frequency)
ff3 = pd.read_csv(os.path.join(wdir, 'FF3_monthly.csv')).dropna().astype({'date_m' : np.int64})


print('Import data - End \n')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#            CALCULATE NUMBER OF RETURN OBSERVATIONS                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Calculate number of valid return observations over a 5-year period.')

# Get number of observations from CRSP data
num_ret = crspm.groupby(['PERMNO', 'date_jun'])['RET'].count().reset_index().rename(columns = {'RET' : 'NUM_RET'})
# Calculate 5-year number number of valid return observations
num_ret['NUM_RET_5Y'] = num_ret.groupby('PERMNO')['NUM_RET'].transform(lambda s: s.rolling(5).sum())
# Merge with firmchars
firmchars = pd.merge(firmchars, num_ret, how = 'left', on = ['PERMNO', 'date_jun'])
# Create a dummy to check if NUM_RET_5Y. 
# If NUM_RET_5Y = 60 months or 5 years, then exists_5Y = 1
firmchars['exists_5Y'] = np.where(firmchars['NUM_RET_5Y'] == 60, 1, 0)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#             ISOLATE THE FORMATION AND HOLDING DATES                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Isolate the formation and holding dates.')

# Define formation and hold dates using CRSP data
# Isolate the formation dates which coincide with date_jun
fdates = pd.DataFrame(data = crspm['date_jun'].drop_duplicates().sort_values().reset_index(drop = True), columns = ['date_jun'])
# Get the holding date 5 years before
fdates['date_hold'] = fdates['date_jun'].shift(4)
# Isolate only the valid formation and holding dates
fdates = fdates.dropna().reset_index(drop = True).astype({'date_hold' : np.int64})
# Drop the last formation date which corresponds to 202206
fdates = fdates.iloc[:-1]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                DANIEL & TITMAN (1997) METHODOLOGY                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Daniel & Titman methodology - Start \n')

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
    fmask1 = firmchars['exists_5Y'] == 1
    fmask2 = firmchars['date_jun'] == formation_date
    firmchars5Y = firmchars[ fmask1 & fmask2 ].copy()
    # Isolate the entities themselves
    entities5Y = set(firmchars5Y['PERMNO'].values)

    # Isolate the return data of the same entities
    cmask1 = hold_date <= crspm['date_jun']
    cmask2 = crspm['date_jun']<= formation_date
    crspm5Y = crspm[ (crspm['PERMNO'].isin(entities5Y) ) & cmask1 & cmask2 ].copy()
    
    # Replace the values of the 'date_jun' column with formation_date so that FFPortfolios() 
    # method will workas intended in the context of the construction of the new portfolios.
    crspm5Y['date_jun'] = formation_date
    
    # Define the PortSort class 
    portchar = ps.PortSort(df = firmchars5Y, 
                           entity_id = 'PERMNO', 
                           time_id = 'date_jun',    
                           save_dir = ff_dir)
    

    
    # ~~~~~~~~~~~~~
    # SMB FACTOR  #
    # ~~~~~~~~~~~~~ 
    
    # Control execution
    # -----------------
    if do_SMB:
    
        print('SMB factor')
        
        # Create the 2 Size portfolios 
        portchar.FFPortfolios(ret_data = crspm5Y, 
                              ret_time_id = 'date_m', 
                              FFcharacteristics = ['ME'], 
                              FFlagged_periods = [0],
                              FFn_portfolios = [2], 
                              FFquantile_filters = [['NYSE', 1]],
                              weight_col = 'CAP',
                              return_col = 'RET',
                              FFsave = False)
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        size_def = {1 : 'S', 2 : 'B'}
        
        
            
        # Isolate the portfolios and rename the columns
        size_p = portchar.FFportfolios.copy().rename(columns = size_def)
        
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
        
        # Create the 2x3 Size and Book-to-Market portfolios 
        portchar.FFPortfolios(ret_data = crspm5Y,
                              ret_time_id = 'date_m',
                              FFcharacteristics = ['ME', 'BtM'],
                              FFlagged_periods = [0, 0],
                              FFn_portfolios = [2, np.array([0, 0.3, 0.7])], 
                              FFquantile_filters = [['NYSE', 1], ['NYSE', 1] ],
                              weight_col = 'CAP', 
                              return_col = 'RET',
                              FFsave = False)
        
                   
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # BtM : 1 = Low, 2 = Neutral, 3 = High
        sizebtm_def = {'1_1' : 'SL', '1_2' : 'SN', '1_3' : 'SH', \
                       '2_1' : 'BL', '2_2' : 'BN', '2_3' : 'BH'}
        
        # Isolate the portfolios and rename the columns
        sizebtm_p = portchar.FFportfolios.copy().rename(columns = sizebtm_def)
        
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
        portchar.FFPortfolios(ret_data = crspm5Y, 
                              ret_time_id = 'date_m', 
                              FFcharacteristics = ['ME', 'OP'],
                              FFlagged_periods = [0, 0], 
                              FFn_portfolios = [2, np.array([0, 0.3, 0.7])], 
                              FFquantile_filters = [['NYSE', 1], ['NYSE', 1] ],
                              weight_col = 'CAP', 
                              return_col = 'RET',
                              FFsave = False)  
    
    
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # OP : 1 = Weak, 2 = Neutral, 3 = Robust
        sizermw_def = {'1_1' : 'SW', '1_2' : 'SN', '1_3' : 'SR', \
                       '2_1' : 'BW', '2_2' : 'BN', '2_3' : 'BR'}
        
        # Isolate the portfolios and rename the columns
        sizermw_p = portchar.FFportfolios.copy().rename(columns = sizermw_def)
    
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
        portchar.FFPortfolios(ret_data = crspm5Y,
                              ret_time_id = 'date_m',
                              FFcharacteristics = ['ME', 'INV'],
                              FFlagged_periods = [0, 0], 
                              FFn_portfolios = [2, np.array([0, 0.3, 0.7])], 
                              FFquantile_filters = [['NYSE', 1], ['NYSE', 1] ],
                              weight_col = 'CAP', 
                              return_col = 'RET',
                              FFsave = False)
    
        
        # Renaming the portfolios as per Fama & French (2015) 
        # Size : 1 = Small, 2 = Big
        # INV : 1 = Conservative, 2 = Neutral, 3 = Aggressive
        sizecma_def = {'1_1' : 'SC', '1_2' : 'SN', '1_3' : 'SA', \
                       '2_1' : 'BC', '2_2' : 'BN', '2_3' : 'BA'}
        
        # Isolate the portfolios and rename the columns
        sizecma_p = portchar.FFportfolios.copy().rename(columns = sizecma_def)
    
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
        crspmkt = pd.merge(crspm5Y, firmchars5Y[['date_jun', 'PERMNO', 'CAP']], on = ['date_jun', 'PERMNO']).dropna()
        
        # Calculate the market return 
        mkt_DT97 = crspmkt.groupby(by = 'date_m').agg( {'RET' : lambda x: WeightedMean(x, df=crspmkt, weights = 'CAP') } )
        # Rename the column
        mkt_DT97.columns = ['MKT_DT97']    
        # Get the excess return of the market portfolio
        mkt_DT97 = mkt_DT97.join(ff3.set_index('date_m')['RF'])
        mkt_DT97['MKT_DT97'] = mkt_DT97['MKT_DT97']  - mkt_DT97['RF']
        
        # Restructure the DataFrame
        mkt_DT97 = mkt_DT97['MKT_DT97'].reset_index()
        mkt_DT97['date_jun'] = formation_date
        # and concat    
        MKT_DT97 = pd.concat([MKT_DT97, mkt_DT97], axis = 0)    
        
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               PUTTING EVERYTHING TOGETHER AND SAVE                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print('Putting everything together.')

# List to save all 
l = []
    
if do_SMB:
    SMB_DT97 =  SMB_DT97.reset_index(drop = True)
    SMB_DT97.to_csv(os.path.join(wdir, 'SMB_DT97_monthly.csv'), index = False)
    print('SMB_DT_97 is saved.')
    l.append(SMB_DT97)

if do_HML:
    HML_DT97 =  HML_DT97.reset_index(drop = True)
    HML_DT97.to_csv(os.path.join(wdir, 'HML_DT97_monthly.csv'), index = False)
    print('HML_DT_97 is saved.')
    l.append(HML_DT97)
    
if do_RMW:
    RMW_DT97 =  RMW_DT97.reset_index(drop = True)
    RMW_DT97.to_csv(os.path.join(wdir, 'RMW_DT97_monthly.csv'), index = False)
    print('RMW_DT_97 is saved.')
    l.append(RMW_DT97)

if do_CMA:
    CMA_DT97 =  CMA_DT97.reset_index(drop = True)
    CMA_DT97.to_csv(os.path.join(wdir, 'CMA_DT97_monthly.csv'), index = False)
    print('CMA_DT_97 is saved.')
    l.append(CMA_DT97)
    
if do_MKT:
    MKT_DT97 =  MKT_DT97.reset_index(drop = True)
    MKT_DT97.to_csv(os.path.join(wdir, 'MKT_DT97_monthly.csv'), index = False)
    print('MKT_DT_97 is saved.') 
    l.append(MKT_DT97)


FF5_DT97 = reduce(lambda a, b : pd.merge(a,b, how = 'inner', on = ['date_m', 'date_jun']), l)
FF5_DT97.to_csv(os.path.join(wdir, 'FF5_DT97_monthly.csv'), index = False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#              FIGURE 1 IN DANIEL & TITMAN (1997) PAPER                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Control execution
if do_figure1:
    
    # Import the FF5_DT97
    FF5_DT97 = pd.read_csv(os.path.join(wdir, 'FF5_DT97_monthly.csv'))
    
    
    # Function that returns the number of months between 2 dates in YYYYmm
    def num_months(startdt, enddt):
        startdt = pd.to_datetime(startdt, format = '%Y%m')
        enddt = pd.to_datetime(enddt, format = '%Y%m')
        return (enddt.year - startdt.year)*12 + enddt.month - startdt.month   
    
    # Define the number of months before the formation date
    FF5_DT97['Months_before'] = - FF5_DT97.apply(lambda x: num_months(x['date_m'], x['date_jun']), axis = 1)


    # For HML 
    # -------
    hml_ret = FF5_DT97.groupby('Months_before')['HML_DT97'].mean()
    
    # Plot 
    plt.figure()
    (100*hml_ret).plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return (%)')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97_HML.png'))
    plt.close()
    
    
    # For SMB
    # -------
    smb_ret = FF5_DT97.groupby('Months_before')['SMB_DT97'].mean()
    
    # Plot 
    plt.figure()
    (100*smb_ret).plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return (%)')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97_SMB.png'))
    plt.close()   
    
    
    
    # For RMW
    # -------
    rmw_ret = FF5_DT97.groupby('Months_before')['RMW_DT97'].mean()
    
    # Plot 
    plt.figure()
    (100*rmw_ret).plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return (%)')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97_RMW.png'))
    plt.close()      
    
    
    # For CMA
    # -------
    cma_ret = FF5_DT97.groupby('Months_before')['CMA_DT97'].mean()
    
    # Plot 
    plt.figure()
    (100*cma_ret).plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return (%)')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97_CMA.png'))
    plt.close()  
    
    
    # For MKT
    # -------
    mkt_ret = FF5_DT97.groupby('Months_before')['MKT_DT97'].mean()
    
    # Plot 
    plt.figure()
    (100*mkt_ret).plot()
    plt.xlabel('Months before the formation date')
    plt.ylabel('Monthly return (%)')
    plt.savefig(os.path.join(wdir, 'Figure_1_DanielTitman97_MKT.png'))
    plt.close()    
    
    
