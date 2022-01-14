# ANGELS-PS<sup>3</sup>
The **AN**alyst of **G**ravity **E**stimation with **L**ow-orbit **S**atellites:<br>
The **P**ost-processing **S**ystem of **S**patial-domain and **S**pectral-domain<br>
----
## Main function
ANGELS-PS3 is a software that integrates various post-processing methods for GRACE data.
### Pre-preprocessing
In data pre-preprocessing, PS3 can carry out: <br>
	(1)low order coefficient replacement; (2)static field deduction.<br>
### Post-processing
ANGELS-PS3 can be divided into two types of post-processing: <br>
spatial post-processing and Spectral-domain post-processing.  <br>
Spectral-domain post-processing includes: <br>
	(1)Mascon algorithm; (2)Slepian Function; (3)Forward Modelling.<br>
Frequency-domain post-processing includes: <br>
	(1)Gaussian filter; (2)Fan filter; (3)Swenson filter; (4)PnMm filter; (5)DDK filter.<br>
### Output
Finally, the equivalent water height or the spherical harmonic coefficient after treatment can be output.
	
## Release time
We will release the code in June 2022.

## Reference
Chen, J. L. ,  Wilson, C. R. ,  Tapley, B. D. , &  Grand, S. . (2007). Grace detects coseismic and postseismic deformation from the sumatra‐andaman earthquake. Geophysical Research Letters, 34(13), 173-180.<br>
Kusche, J. ,  Schmidt, R. ,  Petrovic, S. , &  Rietbroek, R. . (2009). Decorrelated grace time-variable gravity solutions by gfz, and their validation using a hydrological model. Journal of Geodesy, 83(10), 903-913.<br>
Swenson, S. , &  Wahr, J. . (2006). Post-processing removal of correlated errors in grace data. Geophysical Research Letters, 33(8), L08402.<br>
Wahr, J. ,  Molenaar, M. , &  Bryan, F. . (1998). Time variability of the earth's gravity field: hydrological and oceanic effects and their possible detection using grace. Journal of Geophysical Research Solid Earth, 103(B12).<br>
Zhang, Z. Z. ,  Chao, B. F. ,  Lu, Y. , &  Hsu, H. T. . (2009). An effective filtering for grace time-variable gravity: fan filter. Geophysical Research Letters, 36(17), L17311.
