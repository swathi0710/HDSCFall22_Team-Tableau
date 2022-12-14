from flask import Flask
from flask import request
import pandas as pd
import pickle
import xgboost
#from flask import render_template

#load model:


app = Flask(__name__)

@app.route('/')
def homepage():
    website = '''
<!DOCTYPE html>
<html>

<body>
    <style>
      a:link {
      color: #9e0340;
      background-color:#eae7e8;
       }

      a:visited {
      color: #9e0340;
      background-color:#eae7e8;
      }
      body {
        background-image: url("https://raw.githubusercontent.com/swathi0710/website/main/house.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        color:#003333;
      }
    </style>
    <a href="/about"> About |</a> <a href="https://github.com/swathi0710/HDSCFall22_Team-Tableau">GitHub repository</a>
    <br>
    <br>
    <h1><center><img src="https://raw.githubusercontent.com/swathi0710/website/main/title.png"></center></h1>
    <ul>
        <center><h1><a href="/result">Sale Price Estimator</a></h1></center>
    </ul>
    <br>
    <br>
    <br>

</body>
</html>'''
    return website

@app.route('/result')
def my_form():
    website = '''
<!DOCTYPE html>
<html>
<body>
    <style>
      body {
        background-image: url("https://raw.githubusercontent.com/swathi0710/AIML_project_ingredient-analyser/main/gradient.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        color:orange;
        text-align: left;
      }
    </style>
    <center><h1>Estimator</h1></center>
    <center><h2>Enter the features of the house below:</h2>
    <form action="result" method="POST">

        <table><tr><td>Rating of the overall material and finish of the house [1-10]</td><td><input type="number" name="OverallQual" placeholder="0"></td></tr>
        <tr><td>Rates the overall condition of the house [1-5]</td><td><input type="number" name="OverallCond" placeholder="0"></td></tr>
        <tr><td>Year of Original construction </td><td><input type="number" name="YearBuilt" placeholder="0"></td></tr>
        <tr><td>Quality rating of the material on the exterior [1-10]</td><td><input type="number" name="ExterQual" placeholder="0"></td></tr>
        <tr><td>Kitchen Quality [1-10]</td><td><input type="number" name="KitchenQual" placeholder="0"></td></tr>
        <tr><td>Rating of the present condition of the material on the exterior [1-5]</td><td><input type="number" name="ExterCond" placeholder="0"></td></tr>
        <tr><td>Basement Quality [1-5]</td><td><input type="number" name="BsmtQual" placeholder="0"></td></tr>
        <tr><td>Basement Condition [1-5]</td><td><input type="number" name="BsmtCond" placeholder="0"></td></tr>
        <tr><td>Total square feet of basement area</td><td><input type="number" name="TotalBsmtSF" placeholder="0"></td></tr>
        <tr><td>Lot size in square feet</td><td><input type="number" name="LotArea" placeholder="0"></td></tr>
        <tr><td>Remodel year (same as construction year if no remodeling or additions)</td><td><input type="number" name="YearRemodAdd" placeholder="0"></td></tr>
        <tr><td>Type 1 Finished Basement in SqFt </td><td><input type="number" name="BsmtFinSF1" placeholder="0"></td></tr>
        <tr><td>Type 2 Finished Basement in SqFt </td><td><input type="number" name="BsmtFinSF2" placeholder="0"></td></tr>
        <tr><td>Above grade (ground) living area square feet</td><td><input type="number" name="GrLivArea" placeholder="0"></td></tr>
        <tr><td>Low quality finished square feet (all floors)</td><td><input type="number" name="LowQualFinSF" placeholder="0"></td></tr>
        <tr><td>1st Floor area in SqFt</td><td><input type="number" name="1stFlrSF" placeholder="0"></td></tr>
        <tr><td>2nd Floor area in SqFt</td><td><input type="number" name="2ndFlrSF" placeholder="0"></td></tr>
        <tr><td>Healting Quality and Condition [1-5]</td><td><input type="number" name="HeatingQC" placeholder="0"></td></tr>
        <tr><td>Number of Basement full bathrooms</td><td><input type="number" name="BsmtFullBath" placeholder="0"></td></tr>
        <tr><td>Number of Basement half bathrooms</td><td><input type="number" name="BsmtHalfBath" placeholder="0"></td></tr>
        <tr><td>Full bathrooms above grade</td><td><input type="number" name="FullBath" placeholder="0"></td></tr>
        <tr><td>Half bathrooms above grade</td><td><input type="number" name="HalfBath" placeholder="0"></td></tr>
        <tr><td>Unfinished Basement Area in Sqft</td><td><input type="number" name="BsmtUnfSF" placeholder="0"></td></tr>
        <tr><td>Number of Bedrooms above grade (does NOT include basement bedrooms)</td><td><input type="number" name="BedroomAbvGr" placeholder="0"></td></tr>
        <tr><td>Type of dwelling involved in the sale(MSSubClass)</td><td><input type="number" name="MSSubClass" placeholder="0"></td></tr>
        <tr><td>Total rooms above grade (does not include bathrooms)</td><td><input type="number" name="TotRmsAbvGrd" placeholder="0"></td></tr>
        <tr><td>Kitchen above grade </td><td><input type="number" name="KitchenAbvGr" placeholder="0"></td></tr>
        <tr><td>Number of fireplaces </td><td><input type="number" name="Fireplaces" placeholder="0"></td></tr>
        <tr><td>Fireplace Quality[1-5]</td><td><input type="number" name="FireplaceQu" placeholder="0"></td></tr>
        <tr><td>Size of garage in car capacity</td><td><input type="number" name="GarageCars" placeholder="0"></td></tr>
        <tr><td>Size of garage in SqFt</td><td><input type="number" name="GarageArea" placeholder="0"></td></tr>
        <tr><td>Garage quality[1-5]</td><td><input type="number" name="GarageQual" placeholder="0"></td></tr>
        <tr><td>Garage condition[1-5]</td><td><input type="number" name="GarageCond" placeholder="0"></td></tr>
        <tr><td>Wood deck area in square feet</td><td><input type="number" name="WoodDeckSF" placeholder="0"></td></tr>
        <tr><td>Open porch area in square feet</td><td><input type="number" name="OpenPorchSF" placeholder="0"></td></tr>
        <tr><td>Enclosed porch area in square feet</td><td><input type="number" name="EnclosedPorch" placeholder="0"></td></tr>
        <tr><td>Three season porch area in square feet</td><td><input type="number" name="3SsnPorch" placeholder="0"></td></tr>
        <tr><td>Screen porch area in square feet</td><td><input type="number" name="ScreenPorch" placeholder="0"></td></tr>
        <tr><td>Pool area in square feet</td><td><input type="number" name="PoolArea" placeholder="0"></td></tr>
        <tr><td>Value of miscellaneous feature in dollars$</td><td><input type="number" name="MiscVal" placeholder="0"></td></tr>
        <tr><td>Month Sold</td><td><input type="number" name="MoSold" placeholder="0"></td></tr>
        <tr><td>Year Sold</td><td><input type="number" name="YrSold" placeholder="0"></td></tr></table>
        </center>
        <br>
        <center><input type="submit" value="Estimate!"></center>
    </form>
</body>
</html>'''
    return website

@app.route('/about')
def about():
    website = '''
<!DOCTYPE html>
<html>
<body>
    <style>
      body {
        background-image: url("https://raw.githubusercontent.com/swathi0710/AIML_project_ingredient-analyser/main/gradient.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        color:white;
      }
    </style>
    <center><h1><ul>About this Project</ul></h1>
    <p>
    This tool was developed for Stage C of the Hamoye Internship by Team Tableau.
    The tool is part of the final deployment phase of the project which focussed on predicting Sales Prices by houses using the Ames Housing Dataset available on kaggle.
    The most important features which have a high correlation with the predicted value has been chosen to get a predicted Sale Price value with low error.</p>
    </center>
</body>
</html>'''
    return website


@app.route('/result', methods=['POST'])
def my_form_post():

    with open('model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    input1=[]

    column_s=['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
       'YearRemodAdd', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
       'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
       'KitchenQual', 'TotRmsAbvGrd', 'Fireplaces', 'FireplaceQu',
       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold']

    for col in column_s:
        if request.form[col]:
            input1.append(int(request.form[col]))
        else:
            input1.append(0)

    s=pd.DataFrame(input1,index=column_s).T

    r=model.predict(s)
    if not any(input1):
        result=0
    else:
        result=r[0]


    text=[]
    if result<181438.53:
        text.append("below")
    else:
        text.append("above")




    page='''
<!DOCTYPE html>
<html>
<body background="https://raw.githubusercontent.com/swathi0710/AIML_project_ingredient-analyser/main/gradient.jpg" text="orange" alignment=center>
    <br><br><br><br><br><br><br><br><br><br>
    <center><h1>Predicted Price: {} $</h1></center>
    <br>
    <center>This falls {} the average Sale Price for houses in Ames, Iowa.</center>

</body>
</html>'''.format(result,text[0])

    return page




if __name__ == '__main__':
    app.run()