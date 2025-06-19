* Construct baseline ECB shock 

set scheme s2mono

clear all
set more off

/* Directories */
local dirdata  "C:\Users\Utente\Dropbox\Thesis\_Replicate_BRW\Data"   
local diroutput  "C:\Users\Utente\Dropbox\Thesis"  
local slash /
cd "`dirdata'"


/* Set parameters */
local startyear 2003
local endyear 2025 
local diff 1  
local normal=2

// order of my vars: 3mth 6mth 9mth	1yr	2yr	3yr	4yr	5yr	6yr	7yr	8yr 9yr	10yr 11yr 12yr 13yr 14yr 15yr 16yr 17yr 18yr 19yr 20yr 21yr 22yr 23yr 24yr 25yr 26yr 27yr 28yr 29yr 30yr

*local list "01 04 05 08 10 13"  //keeping only 3mth, 1yr, 5yr, 7yr and 10 yr  -----------> FOR ROBUSTNESS CHECKS

local list "01 04 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33" // removing the 2 years yield curve the result improves a lot. 

local n_elements : word count `list'
local shift `n_elements' //number of series above. In order to later align the dataset

local diff_1 = `diff'-1 //shift the diff
local minus 1 //Normalization of the shock
local fomc 1  //set to 1 to make prior to 1994 the following dates of FOMC
local ipo 1 //whether to ipolate the data

local fomcdiff 0 //set to 1 to make prior to 1994 a two day window
if `diff' == 2{ //we need to make the adjustments when 2 day window is used
local fomcdiff = 1
local diff = `diff_1'
di `diff'
di `diff_1'
}


/*  Data Construction */
* Outcome variables
clear
import excel using "Zero-coupon yield_Euro.xlsx", firstrow case(lower)
gen date_s = date(date, "YMD")  
format date_s %tdCCYY-NN-DD   
drop date 
rename date_s date 
     

gen year  = year(date)
gen month = month(date)
gen day   = day(date)
destring year month day, force replace 

if `ipo' == 1 {
    // copy existing Stata date into a scratch var
    gen obdate = date    
    format obdate %tdCCYY-NN-DD

    foreach num of local list {
        rename sveny`num' nsveny`num'
        ipolate nsveny`num' obdate, gen(sveny`num')
        drop nsveny`num'
    }

    drop obdate
}

saveold tempiv.dta,replace


* Policy indicator - 2 year Treasury rates
clear
import excel using "OIS_EA.xlsx", firstrow case(lower)
/* REMOVING THE GAP BETWEEN THE DIFFERENT DATA OIS
gen year= year(observation_date)
gen month=month(observation_date)
gen day=day(observation_date)
gen mdate = ym(year,month)
gen double dgs2_d8= dgs2_d4-0.085 if dgs2_d4!=.&year<2007
replace dgs2_d8=dgs2 if year>=2007
replace new_policy_2=. if new_policy_2==-0.085
replace new_policy_2=. if new_policy_2==0
gen tempo = (new_policy_2 == dgs2_d8)
count if tempo == 0

gen double dgs2_d9= dgs2_d5-0.085 if dgs2_d5!=.&year<2007
replace dgs2_d9=dgs2_d1 if year>=2007

drop j new_policy_2 tempo dgs2_d2 year month day mdate dgs2_d3
rename dgs2_d8 dgs2_d2
rename dgs2_d9 dgs2_d3

export excel using "OIS_EA.xlsx", firstrow(variables) replace
*/
gen year= year(observation_date)
gen month=month(observation_date)
gen day=day(observation_date)
gen mdate = ym(year,month)
drop dgs`normal'_d1-dgs`normal'_d7
format observation_date %tdCCYY-NN-DD
* replace dgs`normal' = "" if dgs`normal' == "ND"
if `ipo' == 1{
 cap: gen ndgs10 = dgs`normal'
 cap: drop dgs10
ipolate ndgs10 observation_date, gen(dgs10)
}


if `fomc' == 1{
gen dgs10_d= dgs10[_n+`diff_1']-dgs10[_n-1] if mdate>=m(1994m2)
replace dgs10_d= dgs10[_n+`diff']-dgs10[_n-`fomcdiff'] if mdate<m(1994m2)
}
else{
gen dgs10_d= dgs10[_n+`diff_1']-dgs10[_n-1]
}
drop mdate

merge 1:1 year month day using tempiv.dta, force
drop _merge
drop if year>`endyear'
drop if year<`startyear'

gen mdate = ym(year,month)
foreach num in  `list' { 
if `fomc' == 1{
gen sveny`num'_d=sveny`num'[_n+`diff_1']-sveny`num'[_n-1] if mdate>=m(1994m2)
replace sveny`num'_d=sveny`num'[_n+`diff']-sveny`num'[_n-`fomcdiff'] if mdate<m(1994m2)
}
else{
gen sveny`num'_d=sveny`num'[_n+`diff_1']-sveny`num'[_n-1]
}
}

drop mdate

drop if year>`endyear'
drop if year<`startyear' 
saveold tempiv.dta, replace
export excel using "C:\Users\Utente\Dropbox\Thesis\_Replicate_BRW\Data\OIS5_d.xlsx", firstrow(variables)


* merge ECB date  
import excel using "N_ECBdate.xlsx", sheet("iv2") firstrow case(lower) clear

replace q=0 if q==.  // q=0 when it is a week prior the meeting
keep date q
gen year= year(date)
gen month=month(date)
gen day=day(date)
gen mdate=ym(year,month)
drop mdate

merge m:1 year month day using tempiv.dta, force
drop if _merge==2
drop _merge


/* Generate IV */
gen iv=dgs10_d if q==1  //Meeting day indicated by q=1
replace iv=-dgs10_d if q==0


/*  Align BRW Shock */
* Step 1: time-series regression to get sensitivity index
tsset date,daily
gen mdate = ym(year, month)

drop if year>`endyear'
drop if year<`startyear'
foreach num in `list' {
ivregress 2sls  sveny`num'_d  (dgs10_d=iv) `setsample'
gen beta`num'_d=_b[dgs10_d]
}
gen aligned_dgs10_d=.
gen sd =. // save standard deviation of BRW shock


* Step 2: cross-section regression to estimate BRW shock
keep if q==1

gen t=_n
qui sum t
local maxt = r(max)
dis `maxt'

forvalues i = 1/`maxt'{  
preserve
 keep if t== `i' 

xpose, clear varname
gen name = substr(_varname,1,4)
gen lastname = substr(_varname,-1,1)
keep if name=="sven" | name=="beta"
drop if lastname != "d"
drop name lastname

gen sveny_d=.
foreach b in `list' {
replace sveny_d=v1 if _varname=="sveny`b'_d"
}


gen beta_d=.
gen beta_d_temp=.
foreach b in `list'{
replace beta_d_temp=v1 if _varname=="beta`b'_d"
}
replace beta_d=beta_d_temp[_n+`shift'] 

drop beta_d_temp
capture {
reg sveny_d  beta_d  if sveny_d!=.
local p=_b[beta_d]
 mat SE=e(V) 
local q = sqrt(SE[1,1])  
} 
restore 
replace aligned_dgs10_d=`p' if t==`i'
replace sd = `q' if t==`i'
}


*  Step 3: renormalize
rena aligned_dgs10_d _newshock
qui reg dgs10_d _newshock
gen scalar_normal = _b[_newshock]
gen newshock1 = _newshock*scalar_normal
drop _newshock
reg dgs10_d newshock1

keep newshock1 date dgs10_d mdate

collapse (sum ) newshock1, by(mdate)
tsset mdate, m
tsfill
replace newshock1 = 0 if missing(newshock1)
rena newshock1 ecb_2
label var ecb_2 "BRW normalized on 5yr interest rate"
tsset mdate,m
saveold "`diroutput'`slash'2yr.dta",replace
export excel using "C:\Users\Utente\Dropbox\Thesis\2yr.xlsx", firstrow(variables) replace datestring("%tmMon_CCYY")


kkk
// MERGING THE JARODISKY SHOCKS AND CHECKING CORRELATION
cd "C:\Users\Utente\Dropbox\Thesis"
import excel using "Jarochinsky MP shocks.xlsx", firstrow case(lower) clear
destring year month, replace
gen mdate = ym(year, month)
format mdate %tm

keep if inrange(mdate, ym(2000,10), ym(2025,10))

merge 1:1 mdate using "NEW.dta"
drop if _merge==2
replace ecb=0 if _merge==1
drop _merge

destring mp_median, ignore(",") replace
corr mp_median ecb 
save "NEW1.dta", replace



import excel "ea_shocks.xlsx", firstrow case(lower) clear
gen mdate = ym(year, month)     
format mdate %tm     

keep if inrange(mdate, ym(2000,10), ym(2025,10))
rename eureon3m_hf shock1 
rename stoxx50_hf shock2
rename deurinflswap2y_d shock3
rename pmnegm_eureon3mstoxx50 shock4
rename pmposm_eureon3mstoxx50 shock5
destring shock1 shock2 shock3 shock4 shock5, replace
keep mdate shock1 shock2 shock3 shock4 shock5
save "Karadi.dta",replace

use "NEW1.dta", clear
merge 1:1 mdate using "Karadi.dta"
drop if _merge==2
drop _merge
save "NEW2.dta", replace



cd "C:\Users\Utente\Dropbox\Thesis\Z.Altavilla\replication_files\data\"
import excel using "dailydataset.xlsx", firstrow case(lower) clear
format date %td
gen year = year(date)
gen month = month(date)
gen mdate = ym(year, month)
format mdate %tm
keep mdate ratefactor1 conffactor1 conffactor2 conffactor3 it2y_conf it2y_rel it10_conf it10_rel it5y_conf it5y_rel
collapse (sum ) ratefactor1 conffactor1 conffactor2 conffactor3 it2y_conf it2y_rel it10_conf it10_rel it5y_conf it5y_rel, by(mdate)
cd "C:\Users\Utente\Dropbox\Thesis"
save "Altavilla.dta", replace


use "NEW2.dta", clear
merge 1:1 mdate using "Altavilla.dta"
drop if _merge==2
drop _merge
label variable ratefactor1  "Target"
label variable conffactor1  "Timing"
label variable conffactor2  "ForwGuidance"
label variable conffactor3  "QE"
corr ratefactor1 conffactor1 conffactor2 conffactor3 ecb 
drop mp mp_normalized
save "NEW3.dta", replace
erase "C:\Users\Utente\Dropbox\Thesis\New1.dta"
erase "C:\Users\Utente\Dropbox\Thesis\New2.dta"


cd "C:\Users\Utente\Dropbox\Thesis"
use "NEW3.dta", clear
order mdate, first
drop mp_pm
corr ecb mp_median shock1 shock4 

*rescaling the factor for comparisons
quietly summarize ecb
local mu_ecb = r(mean)
local sd_ecb = r(sd)

quietly summarize ratefactor1
local mu_rf = r(mean)
local sd_rf = r(sd)
gen ratefactor1_scale = (ratefactor1 - `mu_rf') * (`sd_ecb'/`sd_rf') + `mu_ecb' //the concept is to bring our series to zero, rescale the variance so that it matches the one of our main shock and then move the series at the level of the 'ecb' shock shifting by its mean


quietly summarize conffactor1
local mu_cf = r(mean)
local sd_cf = r(sd)
gen conffactor1_scale = (conffactor1 - `mu_cf') * (`sd_ecb'/`sd_cf') + `mu_ecb'

quietly summarize conffactor2
local mu_cf = r(mean)
local sd_cf = r(sd)
gen conffactor2_scale = (conffactor2 - `mu_cf') * (`sd_ecb'/`sd_cf') + `mu_ecb'

quietly summarize conffactor3
local mu_cf = r(mean)
local sd_cf = r(sd)
gen conffactor3_scale = (conffactor3 - `mu_cf') * (`sd_ecb'/`sd_cf') + `mu_ecb'

drop shock*


*cd "C:\Users\Utente\Dropbox\Thesis"
*export excel using "comparison_ecb.xlsx", firstrow(variables) replace

tsline ecb ratefactor1_scale conffactor1_scale


/* GOOD GRAPH FOR COMPARISON

tsline ecb mp_median shock1 shock4 ratefactor1_scale conffactor1_scale, ///
    name(g1, replace) msymbol(circle) ///
    title("ECB, MP Median & Shocks 1 & 4 Over Time") ///
    xtitle("Date") ytitle("Value") ///
    legend(order(1 "ECB" 2 "MP Median" 3 "Shock1" 4 "Shock4" 5 "Target" 6 "Timing" ) ///
           cols(2) pos(5) ring(0)) ///
    lwidth(medium) scheme(s1color) ///
    xlabel(, grid) ylabel(, grid)


*/

/* FROM GARCH_estimation TO ECB_and_Uncertainty_FinalProxy
cd "C:\Users\Utente\Dropbox\Thesis\Financial Econ Project\Replication folder\Replication codes"
import excel using "GARCH_estimation.xlsx", clear firstrow

// create a new dataset with one obs per month from Oct 2004 to Oct 2023
set obs `=ym(2023,10) - ym(2004,10) + 1'

gen mdate = ym(2004,10) + _n - 1
format mdate %tm
order mdate, first
cd "C:\Users\Utente\Dropbox\Thesis\Monetary Project\Replication folder\Replication codes"
export excel using "ECB_and_Uncertainty_FinalProxy.xlsx", firstrow(variables) replace
*/

* tranforming ECB and Uncertainty series into monthly series fulfilling empty months
clear
cd "C:\Users\Utente\Dropbox\Thesis"
import excel using "ECB_and_Uncertainty_FinalProxy.xlsx", firstrow 
gen mdate_month=mofd(mdate)
format mdate_month %tm
rename mdate v1
rename mdate_month mdate
drop v1
order mdate, first
tsset mdate
tsfill
replace ecb=0 if missing(ecb)
replace GARCHVolatility=0 if missing(GARCHVolatility)
export excel using "ECB_and_Uncertainty_FinalProxy.xlsx", firstrow(variables) replace 






