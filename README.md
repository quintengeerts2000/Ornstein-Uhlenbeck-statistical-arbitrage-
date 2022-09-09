# Ornstein-Uhlenbeck-statistiscal-arbitrage

This repository was based upon a paper by Marco Avallaneda and Jeong-Hyun Lee called Statistical Arbitrage in the US Equities market


1) Decomposing returns based on factor based asset model
----- 

based on the paper by Avalanda and Lee (2008) I will use the following SDE to model the asset returns:
$\frac{dS_i(t)}{S_i(t)}=\alpha_i dt+\displaystyle\sum_{i=1}^N{\beta_{ij}\frac{dI_j(t)}{I_j(t)}}+dX_i(t)$

where the term $\displaystyle\sum_{i=1}^N{\beta_{ij}\frac{dI_j(t)}{I_j(t)}}$ represents the systematic component, in this model the eigenportfolio's are used

the idiosyncratic component is $\alpha_i dt + dX_i(t)$


in this section i will explore ways to find the appropriate Beta's and to isolate the idiosyncratic component of the asset returns

The number of eigenportfolio's I will use to model the systematic component will be chosen such that 55% of the variance can be captured by the eigenportfolio's 

this way i don't use a fixed number of eigenvalues 
------
