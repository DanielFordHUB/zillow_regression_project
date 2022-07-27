# Predicting Tax Valuation

## About the Project
It has become exceedingly important to be able to be able to identify valuable real estate, and properties are flying off the market at an exceptional rate because of the strong buyers market. By analyzing property data from select counties in California, I hope to create a predictive model that can better estimate high value properties on the market.

### Data Dictionary:

   | Column/Feature | Description |
    |--- | --- |
    | __bathrooom__ | The number of bathrooms in the home. |\n
    | __bedrooms__ | The number of bedrooms in the home. |\n
    | __tax_value__ | The tax-assessed value of the home. <br> __Not__ the home's ultimate sale price. |\n
    | __sq_feet__ | The home's square footage. |\n
    | __year_built__ | The year the home was built. |\n
    | __fips__ | \"Federal Information Process System\" code, used to <br> identify zip codes in the U.S. |\n
    | __lot_size__ | The square footage of the lot on which <br> the home is built. |

### Goals

To create a viable predictive model based off of the 2017 Zillow tax valuations


### Initial Questions

1. What are the largest predictiors of tax valuation?

2. How do the number of bedrooms and bathrooms factor into these predictions?

3. Does location matter?

4. How important is square footage?

### Planning

#### To prepare this data i used the following steps

1. Acquire the data using built functions

2. Clean and split data using built functions

3. Use matplotlib, seaborn, and dtale for exploratory data analysis

4. Find possible relational predictors

5. Create baseline ($389831.755)

6. Start modeling using selected features, with OLS, lasso lars, and polynomial models

7. Choose the best model and test (logistic regression)

### How to Reproduce

to reproduce this project you will need to: 

- have a copy of Zillow.csv

- clone this repository

- use the functions in .py files to acquire and clean data

- used libraries are numpy, pandas, matplotlib, seaborn, and sklearn


## Conclusion


## Key Items



## Recomendations



## Next Steps

With more time I would like to gather more data on regional and economic status, as well as test additional models with more features. with this additional time we should hopefully be able to increase our accuracy to an even higher amount.













