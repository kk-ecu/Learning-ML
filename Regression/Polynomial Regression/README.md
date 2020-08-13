Polynomial Regression
===================================

*A Polynomial Regression estimates the relationship between the dependent(y) and independent variable(x) as n<sup>th</sup> degree polynomial. Hence, "The original features are converted into Polunomial features of degree (2, 3 .. ,upto n) and then modeled as multiple linear regression."*

## <samp>y = m<sub>0</sub>x<sub>1</sub><sup>0</sup> + m<sub>1</sub>x<sub>1</sub><sup>1</sup> + m<sub>2</sub>x<sub>1</sub><sup>2</sup> + ... + m<sub>n</sub>x<sub>1</sub><sup>n</sup>


<div align="center"> <img src="linear regression.png"> <img src="polynomial regression.png"> </div>

* `Here the blue line is the ML model`
* <samp>y</samp> = dependent variable `(Salary)`
* <samp>[m<sub>0</sub>,m<sub>1</sub>,..]</samp> = constant
* <samp>[x<sub>0</sub>,x<sub>1</sub>,..]</samp> = original features converted into polynomial features `(Employee position level)`