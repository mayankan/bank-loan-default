# BANK LOAN DEFAULT DATA SCIENCE PROJECT
## INTRODUCTION
### Defaulting on a loan happens when repayments aren't made for a certain period of time. When a loan defaults, it is sent to a debt collection agency whose job is to contact the borrower and receive the unpaid funds. Defaulting will drastically reduce borrower’s credit score, impact his/her ability to receive future credit, and can lead to the seizure of personal property. To disapprove such loan applications who may default in the future, banks need to predict whether a borrower of a loan may default or not. So, they don’t provide the loan in the first place.
## PROBLEM STATEMENT
### I am provided with a bank loan dataset and my goal is to predict whether a given loan may default or not based on the independent features provided in the dataset. The Education Categories which tells about the loan’s borrower’s educational background is a categorical variable which ranges from 1 to 5.  The address of the loan borrower is converted from a geographical location to a unique number which ranges from 0 to 34.
## DATASET
### The dataset provided has 8 independent variables and 1 dependent variable:

### Independent Variables
### 1. age(type: numerical) -> Age in years   Range - 20-56
### 2. ed(type: categorical) -> Education Categories Values - 1,2,3,4,5
### 3. employ(type: numerical) -> Employment Experience in years Range - 0-28
### 4. address(type: numerical) -> Geographical Area converted to number Range - 0-34
### 5. income(type: numerical) -> Income in numbers
### 6. debtinc(type: numerical) -> Debt Income in numbers
### 7. creddebt(type: numerical) -> Debt to Credit ratio
### 8. othdebt(type: numerical) -> Any other debts -->

### Dependent Variable(type: categorical) - default -> 0 = No, 1 = Yes
### Source Code used in the Project is both Python and R. It’s important to install and load essential libraries/packages before running the code. Therefore, just install the following libraries/packages before running the code.

### To run code in Python, following are the packages which needs to be installed followed by the command to be executed on the PowerShell/Terminal before executing the code successfully – 
### 1) numpy – pip install numpy
### 2) pandas – pip install pandas
### 3) matplotlib – pip install matplotlib
### 4) seaborn – pip install seaborn
### 5) pycaret – pip install pycaret
### 6) sklearn – pip install sklearn

### To run the code in R, the libraries can installed within the R script only. If they are already installed, R will updated them and fulfil the requirements to run the code successfully. I have already placed the code to install the packages with the code to load them.
