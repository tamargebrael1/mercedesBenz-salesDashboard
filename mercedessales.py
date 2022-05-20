from click import option
from pyparsing import col
import streamlit as st 
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
from st_btn_select import st_btn_select



@st.cache(suppress_st_warning=True)


def predict_Function(model,year,transmission,mileage,fuelType,tax,mpg,engineSize,predModel,mercedes):
   #The params received in this function represent the user input data
    mercedes=mercedes
    df2 = {'model': model, 'year': year, 'price': 2000, 'transmission': transmission, 'mileage': mileage, 'fuelType': fuelType, 'tax': tax, 'mpg': mpg, 'engineSize': engineSize}
    mercedes = mercedes.append(df2, ignore_index = True) #appending the user input data as the last row in our dataframe
    #Encoding categorical values
    mercedes["transmission"]=mercedes["transmission"].map({'Automatic':0,'Manual':1,'Semi-Auto':2,'Other':3})
    mercedes["fuelType"]=mercedes["fuelType"].map({'Petrol':0,'Hybrid':1,'Diesel':2,'Other':3})
    dum=pd.get_dummies(mercedes["model"],drop_first=True) #Convert categorical variable into dummy/indicator variable
    dum2=pd.DataFrame(dum,columns=dum.columns)
    merc2=pd.concat([mercedes,dum2],axis=1)
    merc2.drop("model",axis=1,inplace=True)
    X=merc2.drop("price",axis=1)
    #y represents the predicted value
    y=merc2["price"]
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0, shuffle=False) #splitting data
    #shuffle = False to make sure that the added params are represented by the last row in the test set
    #Choosing the model depending on the user input
    if predModel == "Linear Regression":
        model=LinearRegression()
    if predModel == "Decision Tree Regression":
        model= DecisionTreeRegressor(random_state=0)
    if predModel == "Random Forest Regression":
        model =RandomForestRegressor(n_estimators = 20, random_state = 0)
    if predModel == "KNN":
        model=KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error=mean_absolute_error(y_test[:-1],y_pred[:-1])  #by [:-1] we're excluding the appended value(last row of the dataset added by the user) from the rmse calculation
    print("Error ---",mean_absolute_error)
    return(y_pred[-1],mean_absolute_error)
with st.sidebar:
    uploaded_file = st.file_uploader("Upload .csv file not exceeding 200 MB")
    selected = option_menu(
        menu_title = None,
        options= ["Home" ,"Data Insights", "Price Prediction" ,"Report"],
        icons = ["house", "bar-chart","graph-up-arrow", "book"],
        menu_icon = "cast",
        default_index =0,
        orientation = "vertical"
)
    display_mode = st.radio('Display mode:',
                                options=('Light Mode','Dark Mode'))
    
if uploaded_file is not None:
     #load the dataframe
    mercedes=pd.read_csv(uploaded_file)
   



    #Adjusting classes depending on the dark mode
if display_mode == 'Dark Mode':
    st.markdown(
        f"""
        <style>
    .css-1v3fvcr {{
        background-color: #16181e;
            }}
        .css-k0sv6k {{
        background-color: #21242d;      
            }}
            .css-1adrfps {{
            background-color: #21242d;     
            }}
            p, ol, ul, dl {{
            color: #f9f9f9 !important;
            }}
            .st-cc {{
            color: #f9f9f9;  
            }}
            .st-el {{
            background-color: #16181e !important;
            }}
            .st-cu {{
            background-color: #16181e !important;
            }}
            .css-9t20qt {{
            background: #ff4b4b !important;
            }}
            .menu .container-xxl[data-v-4323f8ce] {{
            background-color: #16181e !important;
            border-radius: 0px !important;
            }}
            .css-pmxsec .exg6vvm15 {{
            background-color: #21242d;
            }}
            .menu-title[data-v-4323f8ce], .menu .nav-item[data-v-4323f8ce], .menu .nav-link[data-v-4323f8ce], hr[data-v-4323f8ce] {{
            color:#f9f9f9 !important;
            }}
            
            .css-qrbaxs {{
            color: #f9f9f9 !important;
            }}
            .css-1aehpvj {{
            color: #f9f9f9 !important;}}
            .st-cv {{
            background-color: #16181e;
            }}
            .st-fj {{
            background-color: #16181e !important;
            }}
            .st-f4 {{
            background-color: #16181e !important;
            }}
            .st-f6 {{
            background-color: #16181e !important;
            }}
        
        </style>
        """,
        unsafe_allow_html=True
    )
  
if selected == "Data Insights"  :
    if uploaded_file is not None:
        
        navvis = st_btn_select(
        # The different pages
        ('Transmission', 'Fuel type', 'Models'),
        )
        # Display visuals according to the page
        if navvis == 'Transmission':

            if display_mode == 'Dark Mode':
                fig, ax = plt.subplots(figsize = (10,4), dpi = 90,facecolor='#16181e')
            else:
                fig, ax = plt.subplots(figsize = (10,4), dpi = 90)
            Transm = mercedes.transmission.value_counts()
            colors = ['#ff4b4b','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey']
            ax.bar(x = Transm.index, height = Transm.values, color = colors, alpha = 0.9)
            if display_mode == 'Dark Mode':
                ax.set_facecolor('#16181e') 
            # Create labels
            label = Transm.values.tolist()
    
          
            if display_mode == 'Dark Mode':
                ax.text(-1,8000, 'Transmission', {'font': 'serif', 'color': 'white', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
            else:
                ax.text(-1,8000, 'Transmission', {'font': 'serif', 'color': 'black', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
              
            # Text on the top of each bar
            for i in range(len(label)):
                x = i  - 0.35 
                y = (i+18)/2 + label[i]
                x = x-0.08
                y = y + 52
                if display_mode == 'Dark Mode':
                    ax.text(x,y, '{}'.format(Transm.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'white', 'fontsize': 12}, alpha = 0.8)  
                else:
                     ax.text(x,y, '{}'.format(Transm.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'black', 'fontsize': 12}, alpha = 0.8)  
                    

            for loc in ['left','right','top','bottom']:
                ax.spines[loc].set_visible(False) #removing axes
            if display_mode == 'Dark Mode':
                ax.tick_params(axis='x', colors='white')    
                ax.tick_params(axis='y', colors='white')
            else:    
                ax.tick_params(axis='x', colors='black')    
                ax.tick_params(axis='y', colors='black')

            ax.axes.get_yaxis().set_visible(False) #y axis is invisible
            st.pyplot(fig)
            
        if navvis == 'Fuel type':
        
            if display_mode == 'Dark Mode':
                fig, ax = plt.subplots(figsize = (10,4), dpi = 90,facecolor='#16181e')
            else:
                fig, ax = plt.subplots(figsize = (10,4), dpi = 90)
            fuelT = mercedes.fuelType.value_counts()
            colors = ['#ff4b4b','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey']
            ax.bar(x = fuelT.index, height = fuelT.values, color = colors, alpha = 0.9)
            if display_mode == 'Dark Mode':
                ax.set_facecolor('#16181e') 
            # Create labels
            label = fuelT.values.tolist()
            if display_mode == 'Dark Mode':
                ax.text(-1,11000, 'Fuel Type', {'font': 'serif', 'color': 'white', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
            else:
                ax.text(-1,11000, 'Fuel Type', {'font': 'serif', 'color': 'black', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
                
            # Text on the top of each bar
            for i in range(len(label)):
                x = i  - 0.35
                y = (i+18)/2 + label[i]
                x = x-0.08
                y = y + 52
                if display_mode == 'Dark Mode':
                    ax.text(x,y, '{}'.format(fuelT.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'white', 'fontsize': 10}, alpha = 0.8)  
                else:
                    ax.text(x,y, '{}'.format(fuelT.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'black', 'fontsize': 10}, alpha = 0.8)  

            for loc in ['left','right','top','bottom']:
                ax.spines[loc].set_visible(False)
            if display_mode == 'Dark Mode':
                ax.tick_params(axis='x', colors='white')    
                ax.tick_params(axis='y', colors='white')
            else:    
                ax.tick_params(axis='x', colors='black')    
                ax.tick_params(axis='y', colors='black')

            ax.axes.get_yaxis().set_visible(False)
            st.pyplot(fig)
            
        if navvis == 'Models':
            
            if display_mode == 'Dark Mode':
                fig, ax = plt.subplots(figsize = (27,10), dpi = 90,facecolor='#16181e')
            else:
                fig, ax = plt.subplots(figsize = (27,10), dpi = 90)
            ModL = mercedes.model.value_counts()
            colors = ['#ff4b4b','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey','grey']
            ax.bar(x = ModL.index, height = ModL.values, color = colors, alpha = 0.9)
            if display_mode == 'Dark Mode':
                ax.set_facecolor('#16181e') 
            # Create labels
            label = ModL.values.tolist()
            if display_mode == 'Dark Mode':
                ax.text(-1,4000, 'Models', {'font': 'serif', 'color': 'white', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
            else:
                ax.text(-1,4000, 'Models', {'font': 'serif', 'color': 'black', 'fontsize': 18, 'weight':'bold'},alpha = 0.9 )
                
            # Text on the top of each bar
            for i in range(len(label)):
                x = i  - 0.35
                y = (i+18)/2 + label[i]
                x = x-0.08
                y = y + 52
                if display_mode == 'Dark Mode':
                    ax.text(x,y, '{}'.format(ModL.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'white', 'fontsize': 10}, alpha = 0.8)  
                else:
                    ax.text(x,y, '{}'.format(ModL.values[i]),{'font': 'serif', 'weight': 'normal', 'color': 'black', 'fontsize': 10}, alpha = 0.8)  

            for loc in ['left','right','top','bottom']:
                ax.spines[loc].set_visible(False)
            if display_mode == 'Dark Mode':
                ax.tick_params(axis='x', colors='white')    
                ax.tick_params(axis='y', colors='white')
            else:    
                ax.tick_params(axis='x', colors='black')    
                ax.tick_params(axis='y', colors='black')

            ax.axes.get_yaxis().set_visible(False)
            st.pyplot(fig)

if selected == "Home"  :
    if uploaded_file is not None:
        #st.table (mercedes.model.unique())
        option =  st.sidebar.selectbox('Choose a car ID',mercedes.index)
        col1, col2 = st.columns([3, 1])
        #Displaying cars images depending on the model of the selected index
        if  str (mercedes.loc[option,"model"]).strip()== "SLK":
            #SLK
            col1.image(
                        "https://www.motortrend.com/uploads/sites/10/2015/11/2016-mercedes-benz-slkclass-slk350-roadster-angular-front.png",
                        width=400,
                    )
        if  str (mercedes.loc[option,"model"]).strip()== "S Class":
            #S Class
            col1.image("https://www.mbusa.com/content/dam/mb-nafta/us/myco/my22/s/sedan/all-vehicles/2022-S500-SEDAN-AVP-DR.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="SL CLASS":
            #SL Class
            col1.image("https://www.motortrend.com/uploads/sites/10/2015/11/2012-mercedes-benz-slclass-sl550-roadster-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="G Class":
            #G class
            col1.image("https://www.motortrend.com/uploads/sites/10/2019/04/2019-mercedes-benz-g-class-550-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="GLE Class":
            #GLE class
            col1.image("https://www.motortrend.com/uploads/sites/10/2018/03/2018-mercedes-benz-gle-class-coupe-43-amg-4wd-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip() == "GLA Class":
            #GLA class
            col1.image("https://www.motortrend.com/uploads/sites/10/2019/12/2020-mercedes-benz-gla-250-4wd-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()== "A Class":
            #A class
            col1.image("https://img.sm360.ca/ir/w640h390c/images/newcar/ca/2022/mercedes-benz/classe-a-hayon/amg-35-4matic/hatchback/exteriorColors/2022_mercedes-benz_classe-a_amg-35-4matic_hatchback_014_149.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="B Class":
            #B class
            col1.image("https://www.motortrend.com/uploads/sites/10/2016/10/2017-mercedes-benz-b-class-electric-drive-mini-mpv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="GLC Class":
            #GLC class
            col1.image("https://www.motortrend.com/uploads/sites/10/2019/11/2020-mercedes-benz-glc-coupe-glc300-4wd-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="C Class":
            #C class
            col1.image("https://www.motortrend.com/uploads/sites/10/2016/11/2017-mercedes-benz-c-class-300-sport-sedan-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="E Class":
            #E class
            col1.image("https://www.motortrend.com/uploads/sites/10/2019/05/2019-mercedes-benz-e-class-amg-e53-sedan-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="GL Class":
            #GL class
            col1.image("https://www.motortrend.com/uploads/sites/10/2017/11/2016-mercedes-benz-gl-class-450-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="CLS Class":
            #cls class
            col.image("https://www.motortrend.com/uploads/sites/10/2017/09/2018-mercedes-benz-cls-class-amg-63-s-sedan-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="M Class":
            #M class
            col1.image("https://www.motortrend.com/uploads/sites/10/2015/11/2014-mercedes-benz-m-class-ml350-4-matic-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="CLC Class":
            #CLC class
            col1.image("https://www.tyrepowerwonthaggi.com.au/image/vehicle-models/2011/thumb/mercedes-benz-clc-class.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="CLA Class":
            #CLA class
            col1.image("https://www.motortrend.com/uploads/sites/10/2017/02/2017-mercedes-benz-cla-250-sedan-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="V Class":
            #V class
            col1.image("https://www.mercedes-benz-mena.com/en/passengercars/mercedes-benz-cars/models/v-class/v-class-447/_jcr_content/image.MQ6.2.2x.20190826122034.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="CL Class":
            #CL class
            col1.image("https://www.motortrend.com/uploads/sites/10/2015/11/2013-mercedes-benz-cl-class-cl-550-4-matic-coupe-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="GLS Class":
            #GLS 
            col1.image("https://www.motortrend.com/uploads/sites/10/2018/02/2018-mercedes-benz-gls-class-450-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="GLB Class":
            #GLB 
            col1.image("https://www.motortrend.com/uploads/sites/10/2021/12/2022-mercedes-benz-glb-250-suv-angular-front.png?fit=around%7C770:481.25",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="X-CLASS":
            #X-class 
            col1.image("https://www.vansdirect.co.uk/wp-content/uploads/2019/05/x-class_2.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="180":
            #180
            col1.image("https://www.terraecar.com/files/terrae-car_mercedes_c_class.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="CLK":
            #CLK
            col1.image("https://www.hydroflowcarboncleaning.co.uk/wp-content/uploads/2016/11/mersedes-clk.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="R Class":
            #R class
            col1.image("https://www.motortrend.com/uploads/sites/10/2015/11/2012-mercedes-benz-r-class-r350-suv-angular-front.png?fit=around%7C875:492.1875",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="230":
            #230
            col1.image("https://www.pngkey.com/png/full/562-5628271_mercedes-benz-230-sl-pagoda-mercedes-pagoda-png.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="220":
            #220
            col1.image("https://www.seekpng.com/png/full/136-1362770_mercedes-benz-220-s-coupe-50-years-car.png",
                        width=400, 
                    )
        if  str (mercedes.loc[option,"model"]).strip()=="200":
            #200
            col1.image("https://www.pngkey.com/png/full/134-1346428_mercedes-benz-old-png.png",
                        width=400, 
                    )
        #Car info:
        col2.markdown("Model: "+ str ( mercedes.loc[option,"model"]))
        col2.markdown("Year: "+ str (mercedes.loc[option,"year"]))
        col2.markdown("Transmission: "+ str (mercedes.loc[option,"transmission"]))
        col2.markdown("Mileage: "+str (mercedes.loc[option,"mileage"])+" (miles)")
        col2.markdown("Fuel Type: "+ str (mercedes.loc[option,"fuelType"]))
        col2.markdown("Tax: "+str (mercedes.loc[option,"tax"])+" (USD)")
        col2.markdown("miles-per-gallon: "+ str (mercedes.loc[option,"mpg"]))
        col2.markdown("Engine Size: "+ str (mercedes.loc[option,"engineSize"]))
        col2.markdown("Price: "+str (mercedes.loc[option,"price"])+" (USD)")
if selected == "Report":
    if uploaded_file is not None:
        #generate report
        with st.spinner("Generating Report"):
            pr = mercedes.profile_report() 
        st_profile_report(pr) 

if selected == "Price Prediction"  :
    if uploaded_file is not None:
    #st.table (mercedes.model.unique())
        col1, col2 = st.columns(2)
        optionM =  col1.selectbox('Model',mercedes.model.unique())
        optionY =  col2.selectbox('Year',mercedes.year.unique())
        optionT =  col1.selectbox('Transmission Type',mercedes.transmission.unique())
        optionF =  col2.selectbox('Fuel Type',mercedes.fuelType.unique())
        optionMi=  col1.text_input('Mileage',63)
        optionTax= col2.text_input('Tax',32)
        optionmpg= col1.text_input('miles-per-gallon',12)
        optioneng= col2.text_input('Engine Size',1.8)
        optionPredictModel = st.sidebar.selectbox('Kindly choose a prediction model',
        ('Linear Regression', 'Decision Tree Regression', 'Random Forest Regression',"KNN"))
        pred = predict_Function(optionM,optionY,optionT,optionMi,optionF,optionTax,optionmpg,optioneng,optionPredictModel,mercedes)
        st.markdown("**Predicted Price:**")
        st.markdown(pred[0])
        st.markdown("RMSE:")
        st.markdown(pred[1])
    

