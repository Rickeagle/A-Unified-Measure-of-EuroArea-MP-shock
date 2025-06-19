ttt
clear
cd "C:\Users\Utente\Dropbox\Thesis"
import excel using "comparison_ecb.xlsx", firstrow
gen mdate_month=mofd(mdate)
format mdate_month %tm
rename mdate v1
rename mdate_month mdate
drop v1
order mdate, first
tsset mdate


*** Cleaning from the outliers!!
*Done in matlab

*** European Regression!!!
local otherX ""
local opt "bdec(3) label nocons tex nonotes"
cd "C:\Users\Utente\Dropbox\Thesis\Analysis\Tables"
reghdfe ecb ratefactor1_scale conffactor1_scale conffactor2_scale conffactor3_scale `otherX'
su ecb if e(sample)
outreg2 using reg_factor, replace `opt' ctitle(Do the Factors explain my shock?) addstat(Mean dep. var, `r(mean)')

reghdfe ecb_5yr ratefactor1_scale conffactor1_scale conffactor2_scale conffactor3_scale `otherX'
su ecb_5yr if e(sample)
outreg2 using reg_factor, append `opt' addstat(Mean dep. var, `r(mean)')


     
****Comparison with fitted values
reghdfe ecb ratefactor1_scale conffactor1_scale conffactor2_scale conffactor3_scale `otherX'
predict ecb_hat, xb
*scatter ecb_hat ecb

local cut = tm(2018m09)
local cut2 = tm(2017m01)
corr ecb ecb_hat if mdate<=`cut2'
label variable ecb "ECB_2yr"
label var ecb_5yr "ECB_5yr"
label var ecb_hat "Fitted Factors"
local rho = r(rho)
set scheme s2color 
twoway ///
  (line ecb     mdate if mdate<=`cut') ///
  (line ecb_hat mdate if mdate<=`cut'), ///
  xlabel(, nogrid) ///
  ylabel(, nogrid) ///
  legend(on) ///
  note("Corr = `: display %4.2f `rho''")
graph export "FittedValue_Comparison.pdf", as(pdf) replace



****** COMPARISON OF ECB_5yr WITH FITTED VALUES
reghdfe ecb_5yr ratefactor1_scale conffactor1_scale conffactor2_scale conffactor3_scale `otherX'
predict ecb5_hat, xb
*scatter ecb_hat ecb

local cut = tm(2018m09)
local cut2 = tm(2017m01)
corr ecb_5yr ecb5_hat if mdate<=`cut2'
label variable ecb "ECB_2yr"
label var ecb_5yr "ECB_5yr"
label var ecb5_hat "Fitted Factors 5"
local rho = r(rho)
set scheme s2color 
twoway ///
  (line ecb     mdate if mdate<=`cut') ///
  (line ecb_hat mdate if mdate<=`cut'), ///
  xlabel(, nogrid) ///
  ylabel(, nogrid) ///
  legend(on) ///
  note("Corr = `: display %4.2f `rho''")
graph export "FittedValue_Comparison_5.pdf", as(pdf) replace

ren ecb_hat FittedValue_2
ren ecb5_hat FittedValue_5
export excel using "comparison_ecb.xlsx", firstrow(variables) replace


*****FITTING JARO AS WELL:  NOT GOOD RESULTS, HENCE DO NOT GO FORWARD

reghdfe ecb mp_median `otherX'
predict ecb_jaro_hat, xb

local cut = tm(2018m09)
local cut2 = tm(2017m01)
corr ecb ecb_jaro_hat if mdate<=`cut2'
label var ecb_jaro_hat "Fitted Jaro"
local rho = r(rho)
set scheme s2color 
twoway ///
  (line ecb     mdate if mdate<=`cut') ///
  (line ecb_jaro_hat mdate if mdate<=`cut'), ///
  xlabel(, nogrid) ///
  ylabel(, nogrid) ///
  legend(on) ///
  note("Corr = `: display %4.2f `rho''")
graph export "Jaro_FittedValue_Comparison.pdf", as(pdf) replace
ren ecb_jaro_hat FittedValue_Jaro










*********************************************
*** American Regression!!!
gen ecb_lag1=L.ecb
gen ecb_lag2=L2.ecb
gen ecb_lag3=L3.ecb
rename brw_2024_num brw
gen brw_lag1=L.brw 
gen brw_lag2=L2.brw
gen brw_lag3=L3.brw

*** IS THE ECB FOLLOWER?
local otherX ""
local opt "bdec(3) label nocons tex nonotes"
cd "C:\Users\Utente\Dropbox\Thesis\Analysis\Tables"
reghdfe ecb brw `otherX'
su ecb if e(sample)
outreg2 using reg_eu, replace `opt' ctitle(Does the ECB follow the FED?) addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 `otherX'
su ecb if e(sample)
outreg2 using reg_eu, append `opt' addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 brw_lag2 `otherX'
su ecb if e(sample)
outreg2 using reg_eu, append `opt' addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 brw_lag2 brw_lag3 `otherX'
su ecb if e(sample)
outreg2 using reg_eu, append `opt' addstat(Mean dep. var, `r(mean)')
// It appears no clear evidence of the ECB shock on following the USA strategies
local otherX "ecb_lag1 ecb_lag2 ecb_lag3 "
local opt "bdec(3) label nocons tex nonotes"
reghdfe ecb brw `otherX'
su ecb if e(sample)
outreg2 using reg_eu_controls, replace `opt' ctitle(Does the ECB follow the FED?) addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 `otherX'
su ecb if e(sample)
outreg2 using reg_eu_controls, append `opt' addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 brw_lag2 `otherX'
su ecb if e(sample)
outreg2 using reg_eu_controls, append `opt' addstat(Mean dep. var, `r(mean)')
reghdfe ecb brw brw_lag1 brw_lag2 brw_lag3 `otherX'
su ecb if e(sample)
outreg2 using reg_eu_controls, append `opt' addstat(Mean dep. var, `r(mean)')



**** IS THE ECB LEADER?


local otherX ""
local opt "bdec(3) label nocons tex nonotes"
cd "C:\Users\Utente\Dropbox\Thesis\Analysis\Tables"
reghdfe brw ecb `otherX'
su brw if e(sample)
outreg2 using reg_us, replace `opt' ///
    ctitle(Does the FED follow the ECB?) ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 `otherX'
su brw if e(sample)
outreg2 using reg_us, append `opt' ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 ecb_lag2 `otherX'
su brw if e(sample)
outreg2 using reg_us, append `opt' ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 ecb_lag2 ecb_lag3 `otherX'
su brw if e(sample)
outreg2 using reg_us, append `opt' ///
    addstat(Mean dep. var, `r(mean)')


local otherX "brw_lag1 brw_lag2 brw_lag3"
local opt "bdec(3) label nocons tex nonotes"

reghdfe brw ecb `otherX'
su brw if e(sample)
outreg2 using reg_us_controls, replace `opt' ///
    ctitle(Does the FED follow the ECB? – controls) ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 `otherX'
su brw if e(sample)
outreg2 using reg_us_controls, append `opt' ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 ecb_lag2 `otherX'
su brw if e(sample)
outreg2 using reg_us_controls, append `opt' ///
    addstat(Mean dep. var, `r(mean)')

reghdfe brw ecb ecb_lag1 ecb_lag2 ecb_lag3 `otherX'
su brw if e(sample)
outreg2 using reg_us_controls, append `opt' ///
    addstat(Mean dep. var, `r(mean)')


*********************************************************************
	*******CAUSATION ECB AND BRW
	
	
varsoc ecb brw, maxlag(3)

* 3. Estimate reduced‐form VAR with chosen lags (e.g. 2)
var ecb brw, lags(1/3)

* 4. Test Granger causality
vargranger

* 5. Compute orthogonalized IRFs (Cholesky ordering: ecb → brw)
irf set ECB_BRW, replace
irf create irf1, step(12)   // 12‐period horizon

* 6. Plot impulse responses
irf graph oirf, impulse(ecb) response(brw) ///
    title("BRW response to ECB shock")
irf graph oirf, impulse(brw) response(ecb) ///
    title("ECB response to BRW shock")

* 7. Forecast‐error variance decomposition
irf graph fevd, impulse(ecb brw)

* 8. Simple lag‐regressions of structural shocks
reg brw L.brw L.ecb L2.ecb L3.ecb, nocons
test L2.ecb L3.ecb

reg ecb L.ecb L.brw, nocons
test L.brw
****************************************************************












////////////////////// REPLICATING JAROCINSKY AND KARADI PROCEDURE

clear
cd "C:\Users\Utente\Dropbox\Thesis\Analysis"
import excel using "euro_stoxx50.xlsx", firstrow
rename Date date
save "C:\Users\Utente\Dropbox\Thesis\Analysis\euro_stoxx50.dta", replace


clear
cd "C:\Users\Utente\Dropbox\Thesis\Analysis"
import excel using "OIS_EA.xlsx", firstrow
label variable dgs2 "Policy2"
label variable dgs2_d3 "Policy5"
keep observation_date dgs2 dgs2_d3
gen date = observation_date
format date %td
drop observation_date
order date, first
save "C:\Users\Utente\Dropbox\Thesis\Analysis\OIS_EA.dta", replace

**merging Euro OIS and stock market
use "OIS_EA.dta", clear
merge 1:1 date using "euro_stoxx50.dta"
keep if _merge==3
drop _merge
local start = date("1jan2004","DMY")
local end   = date("31dec2024","DMY")
keep if inrange(date, `start', `end')
**ipolating 
gen year  = year(date)
gen month = month(date)
gen day   = day(date)
gen obdate = date    
format obdate %tdCCYY-NN-DD
tsset obdate
tsfill
rename dgs2 ndgs2
rename dgs2_d3 ndgs2_d3
rename STOXX50E nSTOXX50E
ipolate ndgs2 obdate, gen(dgs2)
ipolate ndgs2_d3 obdate, gen(dgs2_d3)
ipolate nSTOXX50E obdate, gen(STOXX50E)
drop date year month day ndgs2 ndgs2_d3 nSTOXX50E
rename obdate date
format date %td
rename dgs2 ois2
rename dgs2_d3 ois5


** generating diff variables and MONETARY DUMMY
cap gen ois2_diff = ois2 - ois2[_n-1]
cap gen ois5_diff = ois5 - ois5[_n-1]
cap gen stoxx50_diff= STOXX50E - STOXX50E[_n-1]

gen monetary_dummy2=.
replace monetary_dummy2=1 if ois2_diff>0&stoxx50_diff<0 
replace monetary_dummy2=1 if ois2_diff<0&stoxx50_diff>0 
replace monetary_dummy2=0 if missing(monetary_dummy2) 

gen monetary_dummy5=.
replace monetary_dummy5=1 if ois5_diff>0&stoxx50_diff<0 
replace monetary_dummy5=1 if ois5_diff<0&stoxx50_diff>0 
replace monetary_dummy5=0 if missing(monetary_dummy5) 

save merged.dta,replace


**Lets import the announcements date, merge it with the merged dataset 
clear
cd "C:\Users\Utente\Dropbox\Thesis\Analysis"
import excel "N_ECBdate.xlsx", sheet("iv2") firstrow case(lower) 
keep if q==1
drop q
format date %td

merge 1:1 date using "merged.dta"
keep if _merge==3
drop _merge
gen mdate=mofd(date)
bysort mdate (date): drop if _n>1
tsset mdate
tsfill
format mdate %tm
replace monetary_dummy2=0 if missing(monetary_dummy2)
replace monetary_dummy5=0 if missing(monetary_dummy5)
label variable ois2_diff "Surprises in the 2 year-OIS"
label variable ois5_diff "Surprises in the 5 year-OIS"
label variable stoxx50_diff "Surprises in the Euro STOXX index"
replace stoxx50_diff=stoxx50_diff/100
replace stoxx50_diff=0 if stoxx50_diff<-3
**** Replicating Jaro and Karadi pag. 24
twoway ///
  (scatter stoxx50_diff ois2_diff, msize(tiny) mcolor(black)) ///
  , xline(0, lpattern(solid) lwidth(medium)) ///
    yline(0, lpattern(solid) lwidth(medium)) ///
    xlabel(, nogrid) ///
    ylabel(, nogrid) ///
    legend(off) ///
    ytitle("Surprises in the Euro STOXX index") ///
    xtitle("Surprises in the 2 year-OIS") ///
    graphregion(color(white)) ///
    name(raw, replace)

graph export "scatter_OIS2.pdf", as(pdf) replace

twoway ///
  (scatter stoxx50_diff ois5_diff, msize(tiny) mcolor(black)) ///
  , xline(0, lpattern(solid) lwidth(medium)) ///
    yline(0, lpattern(solid) lwidth(medium)) ///
    xlabel(, nogrid) ///
    ylabel(, nogrid) ///
    legend(off) ///
    ytitle("Surprises in the Euro STOXX index") ///
    xtitle("Surprises in the 5 year-OIS") ///
    graphregion(color(white)) ///
    name(raw, replace)

graph export "scatter_OIS5.pdf", as(pdf) replace

**creating the monetary policy marks to delete information effect days
keep mdate monetary_dummy2 monetary_dummy5
order mdate, first
export excel using "C:\Users\Utente\Dropbox\Thesis\Analysis\monetary_marks.xlsx", firstrow(variables) replace



** Comparison OIS2_d and OIS5_d when meetings occur
clear
cd "C:\Users\Utente\Dropbox\Thesis\_Replicate_BRW\Data"
import excel using "OIS_diff.xlsx", firstrow
format observation_date %td
rename observation_date date
drop if missing(date)
*gen mdate=mofd(observation_date)
*format mdate %tm
cd "C:\Users\Utente\Dropbox\Thesis\Analysis"
save OIS_diff.dta, replace

clear
cd "C:\Users\Utente\Dropbox\Thesis\Analysis"
import excel "N_ECBdate.xlsx", sheet("iv2") firstrow case(lower) 
keep if q==1
drop q
format date %td

merge 1:1 date using "OIS_diff.dta"
keep if _merge==3
drop _merge
gen mdate=mofd(date)
format mdate %tm
tsset mdate

**Replication figure E.1 Altavilla
regress OIS2_d OIS5_d
local b: display %4.2f _b[OIS5_d]
local t: display %4.2f _b[OIS5_d]/_se[OIS5_d]

twoway ///
  (scatter OIS2_d OIS5_d, mcolor(black) msymbol(circle)) ///
  (lfit   OIS2_d OIS5_d, lcolor(gray) lpattern(solid)) , ///
  xtitle("OIS5_d 1-day changes") ///
  ytitle("OIS2_d 1-day changes") ///
  legend(off) ///
  note("β = `b' (t = `t')", position(11))
graph export "OIS_scatter.pdf", as(pdf) replace




***** DESCRIPTIVES
* 1. Compute scalars for OIS5_d
quietly summarize OIS5_d
scalar min5        = r(min)
scalar max5        = r(max)
scalar mean5       = r(mean)
scalar sd5         = r(sd)
quietly summarize date if OIS5_d==min5
scalar date_min5   = r(min)
quietly summarize date if OIS5_d==max5
scalar date_max5   = r(min)
* 2. Compute scalars for OIS2_d
quietly summarize OIS2_d
scalar min2        = r(min)
scalar max2        = r(max)
scalar mean2       = r(mean)
scalar sd2         = r(sd)
quietly summarize date if OIS2_d==min2
scalar date_min2   = r(min)
quietly summarize date if OIS2_d==max2
scalar date_max2   = r(min)
* 3. Build a small table in memory and display it
clear
set obs 6
gen str12 stat    = ""
gen str12 OIS5    = ""
gen str12 OIS2    = ""
replace stat = "Min"         in 1
replace OIS5 = string(min5, "%9.3f")   in 1
replace OIS2 = string(min2, "%9.3f")   in 1
replace stat = "Date of Min" in 2
replace OIS5 = string(date_min5, "%td") in 2
replace OIS2 = string(date_min2, "%td") in 2
replace stat = "Mean"        in 3
replace OIS5 = string(mean5, "%9.3f")  in 3
replace OIS2 = string(mean2, "%9.3f")  in 3
replace stat = "SD"          in 4
replace OIS5 = string(sd5, "%9.3f")    in 4
replace OIS2 = string(sd2, "%9.3f")    in 4
replace stat = "Date of Max" in 5
replace OIS5 = string(date_max5, "%td") in 5
replace OIS2 = string(date_max2, "%td") in 5
replace stat = "Max"         in 6
replace OIS5 = string(max5, "%9.3f")   in 6
replace OIS2 = string(max2, "%9.3f")   in 6
list stat OIS5 OIS2, noobs clean



** DESCRIPTIVES MACRO_VARS
clear
cd "C:\Users\Utente\Dropbox\Thesis\macroData\NewData"
import excel using "New_MacroData.xlsx", firstrow

tabstat IP HICP_All_EA IntGoodsValue EuroSTOXX50 ExcRate Unempl_Total, statistics(min mean sd max) columns(statistics)














