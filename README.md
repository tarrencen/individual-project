# Individual Project: Nifty 100 Index Closing Price Regression

Goal: Develop an ML regression model to attempt to predict the closing price for the Nifty 100 Index, a listing of 100 stocks traded on India's National Stock Exchange.

	Company Name
0	ACC Ltd.
1	Adani Enterprises Ltd.
2	Adani Green Energy Ltd.
3	Adani Ports and Special Economic Zone Ltd.
4	Adani Transmission Ltd.
5	Ambuja Cements Ltd.
6	Apollo Hospitals Enterprise Ltd.
7	Asian Paints Ltd.
8	Avenue Supermarts Ltd.
9	Axis Bank Ltd.
10	Bajaj Auto Ltd.
11	Bajaj Finance Ltd.
12	Bajaj Finserv Ltd.
13	Bajaj Holdings & Investment Ltd.
14	Bandhan Bank Ltd.
15	Bank of Baroda
16	Berger Paints India Ltd.
17	Bharat Petroleum Corporation Ltd.
18	Bharti Airtel Ltd.
19	Biocon Ltd.
20	Bosch Ltd.
21	Britannia Industries Ltd.
22	Cholamandalam Investment and Finance Company Ltd.
23	Cipla Ltd.
24	Coal India Ltd.
25	Colgate Palmolive (India) Ltd.
26	DLF Ltd.
27	Dabur India Ltd.
28	Divi's Laboratories Ltd.
29	Dr. Reddy's Laboratories Ltd.
30	Eicher Motors Ltd.
31	FSN E-Commerce Ventures Ltd.
32	GAIL (India) Ltd.
33	Gland Pharma Ltd.
34	Godrej Consumer Products Ltd.
35	Grasim Industries Ltd.
36	HCL Technologies Ltd.
37	HDFC Asset Management Company Ltd.
38	HDFC Bank Ltd.
39	HDFC Life Insurance Company Ltd.
40	Havells India Ltd.
41	Hero MotoCorp Ltd.
42	Hindalco Industries Ltd.
43	Hindustan Unilever Ltd.
44	Housing Development Finance Corporation Ltd.
45	ICICI Bank Ltd.
46	ICICI Lombard General Insurance Company Ltd.
47	ICICI Prudential Life Insurance Company Ltd.
48	ITC Ltd.
49	Indian Oil Corporation Ltd.
50	Indus Towers Ltd.
51	IndusInd Bank Ltd.
52	Info Edge (India) Ltd.
53	Infosys Ltd.
54	InterGlobe Aviation Ltd.
55	JSW Steel Ltd.
56	Jubilant Foodworks Ltd.
57	Kotak Mahindra Bank Ltd.
58	Larsen & Toubro Infotech Ltd.
59	Larsen & Toubro Ltd.
60	Lupin Ltd.
61	Mahindra & Mahindra Ltd.
62	Marico Ltd.
63	Maruti Suzuki India Ltd.
64	MindTree Ltd.
65	Muthoot Finance Ltd.
66	NMDC Ltd.
67	NTPC Ltd.
68	Nestle India Ltd.
69	Oil & Natural Gas Corporation Ltd.
70	One 97 Communications Ltd.
71	PI Industries Ltd.
72	Pidilite Industries Ltd.
73	Piramal Enterprises Ltd.
74	Power Grid Corporation of India Ltd.
75	Procter & Gamble Hygiene & Health Care Ltd.
76	Punjab National Bank
77	Reliance Industries Ltd.
78	SBI Cards and Payment Services Ltd.
79	SBI Life Insurance Company Ltd.
80	SRF Ltd.
81	Shree Cement Ltd.
82	Siemens Ltd.
83	State Bank of India
84	Steel Authority of India Ltd.
85	Sun Pharmaceutical Industries Ltd.
86	Tata Consultancy Services Ltd.
87	Tata Consumer Products Ltd.
88	Tata Motors Ltd.
89	Tata Steel Ltd.
90	Tech Mahindra Ltd.
91	Titan Company Ltd.
92	Torrent Pharmaceuticals Ltd.
93	UPL Ltd.
94	UltraTech Cement Ltd.
95	United Spirits Ltd.
96	Vedanta Ltd.
97	Wipro Ltd.
98	Zomato Ltd.
99	Zydus Lifesciences Ltd.


### Data Dictionary:

Date - Date in format (DD/MM/YYYY)
Time - Timestamp (HH:MM:SS) [24 Hours Time Format]
Open - Open Price of One minute Candle (a commonly used type of marker on a stock price graph)
High - High Price of One minute Candle
Low - Low Price of One minute Candle
Close - Close Price of One minute Candle

The data's timeframe was the period spanning Feb 2017-Dec 2019.

### Procedure:
- Acquire: Download the data, posted on Kaggle (https://www.kaggle.com/datasets/ishantj/nifty100?select=Nifty_100.csv), and read it into Python (with Pandas library's read_csv function).
- Wrangle: Combine Date and Time columns into one column, convert it to datetime, and set it as the index. 
- Prepare: Engineer a mid-range column by adding High and Low columns together and dividing the sum by two. Partition the data by percentage into train (50%), validate(30%), and test (%20) splits. Make scaled copies of splits. Isolate the target variable ('Close') from the splits.
- Explore: Visualize distributions of each variable, plot their interactions with each other on the train split, and perform statistical testing for possible relationships.
- Model: Initialize and fit LinearRegression, LassoLars, and TweedieRegressor models to scaled train split (target removed) and isolated target (X and y splits), get predictions from models, and evaluate their root mean squared error (RMSE) against a baseline variable's RMSE. Repeat on validate splits. Select best performing models and evaluate them on test split. Select the best model and report its performance. 

### Summary:

- Univariate analysis revealed that the features' distributions were mostly normal.
- After the data was wrangled, the remaining features of the dataset were all related to price. As such, the features were all strongly correlated with each other, confirmed with a combination of visualizations and correlations testing.
- Because of the data's natural linearity, linear machine learning models were deemed most suited to the task of predicting its most probable future values. OLS, LASSO + LARS, and GLM models were trained on and tested against the data.

# Conclusions:
- An OLS Regression model using opening, high, mid-range, and low prices predicts closing price for India's National Stock Exchange Nifty 100 index for dates after 12-31-2019 to within 1.37 Indian rupees.
- The lack of categorical variables or any other variables that were not directly related to price made it impossible to add any context or draw conclusions about why the index's price is what it is and/or why the model predictions are what they are.
- Some very creative feature engineering or an aggregation of other features matched to the data might improve the predictive power of this model.