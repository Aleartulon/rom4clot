# Introduction
This project shows how it is possible to construct a Reduced Order Model (ROM) of blood clot simulations (previously performed) combining Proper Orthogonal Decomposition (POD) and Deep Learning (DL), the POD-DL-ROM method, as it is done in [1]. The blood clot simulations are obtained with the OpenFoam software, which solves a set of parametrized nonlinear PDEs, as explained in [2],[3]. For every simulation we vary the parameters **μ** and we collect the snapshots **u**(t,**μ**) of different biochemical species, varying both in time and parameters. We then apply POD to those data and collect the POD coefficients **u_N**(t,**μ**) of every snapshot. We are thus able to do the mapping (t,**μ**) -> **u_N**(t,**μ**) for the simulations data, and we want to generalize it to unseen vectors (t,**μ**) using a Convolutional AutoEncoder (CAE) and a Deep FeedForward Neural Network(DFNN) as in [1].  
POD poses some limitations, as it represents well big variations in time and parameters, while it gives poorer representaion of small flows (that can appear in time and/or parameters). For this reason we show how an initial clustering can be performed in order to construct more than one POD basis withouth changing the DL structure. To do this we also need to construct a Classifier, which is able to tell to which cluster an (unseen) set of coefficients **u_N**(t,**μ**) belongs.  
Of course the code here presented is not restricted to blood clot simulations.  
The language used is Python 3.9.12 and the DL architecture is written using the API Keras of Tensorflow.

## Directories
[POD-DL-ROM](https://github.com/Aleartulon/rom4clot/tree/main/POD-DL-ROM%20): The code for POD-DL-ROM as in [1] is constructed, for data preparation, training and testing.  
LOCALIZED-POD-DL-ROM: The code for POD-DL-ROM is left unchanged, but we perform an a priori clustering which divides the snapshots in clusters, highlighting the different regimes in time/parameters. It also contains the code for the Classifier.

## References
[[1](https://www.sciencedirect.com/science/article/pii/S0045782521005120)] S. Fresca and A. Manzoni, “POD-DL-ROM: Enhancing deep learning-based reduced order models for nonlinear parametrized PDEs by proper orthogonal decomposition,” Computer Methods in Applied Mechanics and Engineering, 2022.  
[[2](https://hal.archives-ouvertes.fr/hal-03445613/)] R. M. Rojano, D. Lucor, and et al., “Uncertainty quantification of a thrombosis model considering the clotting assay pfa-100®,” 2021.  
[[3](https://www.nature.com/articles/srep42720)] W. Wu, M. Jamiolkowski, W. Wagner, and et al., “Multi-constituent simulation of thrombus deposition” 2017.
## Contact
For more information contact Alessandro Longhi at ale.longhi@studenti.polito.it

