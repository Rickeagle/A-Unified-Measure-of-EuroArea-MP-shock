clear
/*HICP CREATION
cd "C:\Users\Utente\Dropbox\Thesis\macroData"
import excel using "MacroData.xlsx", firstrow clear 
gen mdate = monthly(date, "MY") 
replace mdate = ym(2024,11) if missing(mdate)
format mdate %tmMon_CCYY 
order mdate, first 
drop date
foreach var of varlist ///
    HICP_All_EA HICP_Core_EA HICP_Food_EA HICP_Gas_EA ///
    HICP_All_Ita HICP_Core_Ita HICP_Food_Ita HICP_Gas_Ita {
    replace `var' = log(`var')
}
foreach var of varlist ///
    HICP_All_EA HICP_Core_EA HICP_Food_EA HICP_Gas_EA ///
    HICP_All_Ita HICP_Core_Ita HICP_Food_Ita HICP_Gas_Ita {
    gen `var'_d = `var' - `var'[_n-1]
}
drop HICP_All_EA HICP_Core_EA HICP_Food_EA HICP_Gas_EA HICP_All_Ita HICP_Core_Ita HICP_Food_Ita HICP_Gas_Ita
rename *_d *
export excel using "MacroData.xlsx", firstrow(variables) replace
*/



/* IP
clear
import excel using "IP.xlsx", firstrow clear
gen mdate = monthly(Date, "MY") 
format mdate %tmMon_CCYY
order mdate, first
drop Date
tsset mdate
*tsline IndSurvey
replace IndProd=log(IndProd)
gen IndProd_d=IndProd-IndProd[_n-1]
drop IndProd
rename *_d *
*tsline IndProd
export excel using "IP.xlsx", firstrow(variables) replace
*/

/*LAST VARIABLES FOR LOG-DIFF
clear
import excel using "EuroSTOXX50.xlsx", firstrow clear
gen mdate=monthly(Date,"MY")
format mdate %tmMon_CCYY
order mdate, first 
drop Date
foreach var of varlist ///
    EuroSTOXX50 CostBorrowing IntGoodsValue {
    replace `var' = log(`var')
    gen `var'_d = `var' - `var'[_n-1]
	drop `var'
}
rename *_d *
export excel using "EuroSTOXX.xlsx", firstrow(variables) replace
*/

/* INTERPOLATION SURVEY OF FORECASTERS
clear
import excel using "Forecast INFL.xlsx", firstrow clear
gen mdate = monthly(Date,"MY")    
format mdate %tmMon_CCYY
tsset mdate, monthly
order mdate, first
drop Date
ipolate Forcast_Infl mdate, gen(Forecast_Infl)
drop Forcast_Infl
export excel using "Forecast_Infl.xlsx", firstrow(variables) replace



clear
import excel using "Forecast GDP.xlsx", firstrow clear
gen mdate = ym(real(Date), 1)
format mdate %tmMon_CCYY
tsset mdate, monthly
tsfill
order mdate, first
drop Date
ipolate Forecast_GDP mdate, gen(Forecast_GDP_i)
drop Forecast_GDP
rename Forecast_GDP_i Forecast_GDP
export excel using "Forecast_GDP.xlsx", firstrow(variables) replace
*/

/* GDP 
clear
import excel using "GDP.xlsx", firstrow clear
gen qdate = quarterly(Date, "YQ")
format qdate %tq
gen mdate = qdate * 3  
format mdate %tmMon_CCYY
tsset mdate, monthly
order mdate, first
drop Date qdate
tsfill
ipolate GDP_ea mdate, gen(GDP_EA)
drop GDP_ea
ipolate GDP_italy mdate, gen(GDP_IT)
drop GDP_italy
export excel using "GDP.xlsx", firstrow(variables) replace
*/

clear
/*
import excel using "CPIs.xlsx", firstrow
keep Time *_EA 
gen mdate = monthly(Time, "YM")
format mdate %tm
tsset mdate
gen logHICP=log(HICP_All_EA)
tsline logHICP
*/
clear
/*
cd "C:\Users\Utente\Dropbox\Thesis\_Replicate_BRW\Data\"
import excel using "OIS_EA", firstrow
keep observation_date dgs2 dgs2_d3
gen mdate = mofd(observation_date)
format mdate %tmMon_CCYY
collapse (sum) monthly_OIS2=dgs2 (sum) monthly_OIS5=dgs2_d3, by(mdate)
local start = ym(2004,10)
local stop  = ym(2024,12)
keep if inrange(mdate, `start', `stop')
tsset mdate
cd "C:\Users\Utente\Dropbox\Thesis\macroData"
export excel using "monthly_OIS.xlsx", firstrow(variables) replace
*/

clear
cd "C:\Users\Utente\Dropbox\Thesis\macroData\NewData"
import excel using "New_MacroData.xlsx", firstrow
gen mdate = monthly(Date, "MY") 
format mdate %tm
order mdate, first 
drop Date
foreach var of varlist ///
    IP HICP_All_EA IntGoodsValue EuroSTOXX50 Unempl_Total ExcRate {
    replace `var' = 100*(log(`var'))
}
order Unempl_Total, last
export excel using "100Log_MacroData.xlsx", firstrow(variables) replace




