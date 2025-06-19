# A-Unified-Measure-of-EuroArea-MP-shock
This repository contains the full replication package for the creation of a unified measure of Monetary Policy shock in the Euro Area devoided of the Information Effect.
The paper is here available in .pdf format.

The Replication folder contains 4 subfolders:
1) Proxy Creation: Stata .do file to replicate the shock series for the Euro Area. The relative dataset needed are in the omonimous folder.
2) Volatility analysis and shock comparison: There is the _MainCode.m file containing the financial analysis and the code for the comparison of my proxy with other relevant measures in the literature. The dataset recalled in the code are in the respective folder. Other code files are function called in the MainCode file.
3) Proxy-SVAR and Uncertainty: There is the _MainCode.m file containing the Monetary Economics application. The dataset needed in the code are in the respective folder. Other code files are function crealled in the MainCode file.
4) Other Analysis + Data derivation: Here there are the 'Analysis' and the 'macroData' folder. The former replicates the analysis reported in Appendix B, C and Par.5 of the Paper. In this STATA file there is also the replication of Jarocinski and Karadi procedure. The data recalled in the file are in the omonimous folder.
   The latter contains the creation of the Macro Dataset used in the structural approaches and the sub-sequent 100*log(.) transformation of the variables.
