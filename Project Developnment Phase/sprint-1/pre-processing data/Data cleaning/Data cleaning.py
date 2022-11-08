#*HANDLING MISSING VALUES*

#after looking at the head of the dataset we have NaN and missing values
#To find the of missing values in each column 
#if present it shows true otherwise it shows false
data.isna().any()

#To find the count of missing values each column using sum function
data.isnull().sum()

#Finding the description of the dataset using describe function like mean,median etc.,
data.describe()

#Finding the mode of vehicleType column using mode function
data['vehicleType'].mode()

#total value_counts in vehicleType column
data['vehicleType'].value_counts()

#Replacing all NaN values in vehicleType column using mode 
data['vehicleType'].fillna("limousine",inplace=True)

#Finding the mode of vehicleType column using mode function
data['gearbox'].mode()

#Replacing all NaN values in gearbox column using mode 
data['gearbox'].fillna("manuell",inplace=True)

#Finding the mode of model column using mode function
data['model'].mode()

#Finding the mode of model column using mode function
data['model'].mode()

#Replacing all NaN values in model column using mode 
data['model'].fillna("golf",inplace=True)

#Finding the mode of fueltype column using mode function
data['fuelType'].mode()

#Replacing all NaN values in model column using mode 
data['fuelType'].fillna("benzin",inplace=True)

#Finding the mode of notRepairedDamage column using mode function
data['notRepairedDamage'].mode()

#Replacing all NaN values in notRepairedDamage column using mode 
data['notRepairedDamage'].fillna("nein",inplace=True)
data.head()

#**OUTLIERS DETECTION AND REPLACING OUTLIERS**
sns.boxplot(data['price'])
#finding the interquartilerange of price column
q1=data['price'].quantile(0.25)
q3=data['price'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr#replacing the outliers of price column with mean
data['price']=np.where(data['price']>upper_bound,upper_bound,np.where(data['price']<lower_bound,upper_bound,data['price']))
#boxplot for price column
sns.boxplot(data['price'])#finding the interquartilerange of kilometer column and replacing the outliers with mean 
q1=data['kilometer'].quantile(0.25)
q3=data['kilometer'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
data['kilometer']=np.where(data['kilometer']>upper_bound,data['kilometer'].mean(),np.where(data['kilometer']<lower_bound,data['kilometer'].mean(),data['kilometer']))
#boxplot for kilometer column
sns.boxplot(data['kilometer'])#finding the interquartilerange of powerPS column and replacing the outliers with lower_bound,upper_bound
q1=data['powerPS'].quantile(0.25)
q3=data['powerPS'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
data['powerPS']=np.where(data['powerPS']>upper_bound,upper_bound,np.where(data['powerPS']<lower_bound,lower_bound,data['powerPS']))
#boxplot for powerPS column
sns.boxplot(data['powerPS'])
#finding the interquartilerange of yearOfRegistration column and replacing the outliers with mean 
q1=data['yearOfRegistration'].quantile(0.25)
q3=data['yearOfRegistration'].quantile(0.75)
iqr=q3-q1
lower_bound=q1-1.5*iqr
upper_bound=q3+1.5*iqr
data['yearOfRegistration']=np.where(data['yearOfRegistration']>upper_bound,data['yearOfRegistration'].mode(),np.where(data['yearOfRegistration']<lower_bound,data['yearOfRegistration'].mode(),data['yearOfRegistration']))
#boxplot for yearOfRegistration column
sns.boxplot(data['yearOfRegistration'])
#boxplot for monthOfRegistation column
sns.boxplot(data['monthOfRegistration'])
#Reading the first five rows of cleaned dataset using head function
data.head()
#Exploring Categorical Features
#list of all categorical columns
list(data.select_dtypes('object'))
data['seller'].value_counts()
#counting public and gewerblich types in seller column using countplot
sns.countplot(data['seller'],palette='coolwarm',saturation=0.9)
data['abtest'].value_counts()
#counting the percentage of different types in abtest column using pie chart
plt.pie(data['abtest'].value_counts(),startangle=90,labels=['test','control'],shadow=True,autopct='%1.2f%%')
plt.legend()
plt.title("abtest")
data['offerType'].value_counts()
#counting angebot and gesuch types in offerType column using countplot
sns.countplot(data['offerType'],palette='spring')
data['vehicleType'].value_counts()
#count of each type in vehicleType column
sns.countplot(data['vehicleType'])
#count of each type in gearbox column
sns.countplot(data['gearbox'],palette='pastel')
data['model'].value_counts()
#top 10 models in model column
plt.figure(figsize =(15,6))
sns.countplot(data['model'].value_counts().head(10))
data['fuelType'].value_counts()
plt.figure(figsize =(15,6))
sns.countplot(data['fuelType'])
data['brand'].value_counts().head()
#count of eaach brand in brand column
plt.figure(figsize =(10,6))
sns.countplot(data['brand'])
data['notRepairedDamage'].value_counts()
sns.countplot(data['notRepairedDamage'],palette='spring')
a=list(data.select_dtypes('number'))
for i in a:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    feature = data[i]
    feature.hist(bins=50, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)    
    ax.set_title(i)
plt.show()
#correlation of  dataset using correaltion function 
correlation=data.corr()
correlation
#exploring the correlation using heatmap 
plt.figure(figsize=(15,10))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
#SELLER PRICE
plt.figure(figsize=(8,4))
sns.barplot(x='seller',y='price',data=data,palette='dark')
#VICLETYPE VS PRICE
plt.figure(figsize=(8,6))
sns.barplot(x='vehicleType',y='price',data=data,ci=100,capsize=0.3,saturation=0.8)
#MODEL VS PRICE
plt.figure(figsize=(15,5))
sns.barplot(x='model',y='price',data=data)
#KILOMETER VS PRICE
sns.kdeplot(x='kilometer',y='price',data=data,palette='husl')
#BRAND VS PRICE
plt.figure(figsize=(25,5))
sns.barplot(x='brand',y='price',data=data)
#YEAR OF REGISTRATION VS PRICE
plt.figure(figsize=(15,5))
sns.stripplot(x='yearOfRegistration',y='price',data=data)
#FUEL TYPE VS PRICE
sns.barplot(x='fuelType',y='price',data=data)
#GEARBOX VS KILOMETER
sns.pointplot(x='gearbox',y='kilometer',hue='fuelType',data=data,ci=99,saturation=0.8,capsize=0.3)
#KILOMETER VS PRICE
sns.scatterplot(x='fuelType',y='kilometer',data=data)
#DISTRIBUTION PLOT
#examing the distribution of price column using distplot in seaborn library
plt.figure(figsize=(15,5))
sns.distplot(data['price'])
parameters={'seller':{'privat':0,'gewerblich':1},
            'abtest':{'test':0,'control':1},
            'notRepairedDamage':{'nein':0,'ja':1},
            'vehicleType':{'limousine':0,'kleinwagen':1,'kombi':2,'bus':3,'cabrio':4,'coupe':5,'suv':6,'andere':7},
            'fuelType':{'benzin':0,'diesel':1,'lpg':2,'cng':3 ,'hybrid':4,'andere':5,'elektro':6}}
data_df=data.replace(parameters)
data_df.head()
#converting all catogorical columns into numerical columns using get_dummies function
Fe_df_cleaned=pd.get_dummies(data_df,columns=['offerType','gearbox'],drop_first=True)
Fe_df_cleaned.head()
#shape of the dataset after label encoding
Fe_df_cleaned.shape
Fe_df_cleaned.columns
#removing unncessary columns in the dataset
main_df=Fe_df_cleaned.drop(columns=['dateCrawled','dateCreated','name','lastSeen','brand','model'],axis=1)
main_df.head()
#multivariate analysis
plt.figure(figsize=(15,5))
sns.pairplot(data)
#dividing the dataset into dependent and independent feature
Independent=main_df.drop(['price'],axis=1)
Dependent=main_df['price']
Independent.head()
Dependent.head()


