# Spheroidal harmonics (SOH): Morphological analysis of closed surfaces via the spheroidal harmonic basis functions
## Authors: Mahmoud S. M. Shaqfa and Wim M. van Rees (MIT--MechE)

The code can be used to analyse a given 3D triangle mesh using spheroidal harmonic basis functions. To do so, the points are mapped onto a spheroidal basis using either radial or hyperbolic mapping techniques. Afterwards, we use a fast least-squares projection approach to compute the basis coefficients up to a user-specified maximum degree. The surface can then be analysed and/or reconstructed from the basis coefficients and functions. Compared to existing methods based on traditional spherical harmonics, the mapping onto a spheroidal geometric basis offers an additional parameter (aspect ratio for radial mapping, or eccentricity for hyperbolic mapping) that can be tuned to reduce mapping distortions. This suppress high-frequency oscillations in the reconstructed shapes. Our code includes automatic algorithms to select the mapping parameter for each shape, as specified in the attached paper. Potential applications are envisioned for analysis of closed surfaces in granular particles, heterogeneous materials, and medical applications.

This work is part of the paper: [Spheroidal harmonics for generalizing the morphological decomposition of closed parametric surfaces](https://arxiv.org/abs/2407.03350)

To replicate the results of the paper, please run the python3 notebook: ```spheroidal_harmonics_demo.ipynb```


## Dependencies

To install the required dependencies, you can use the following command:

```bash
pip3 install numpy matplotlib scipy libigl scikit-learn cvxpy h5py mat73
```



## Example output
### The reconstruction of [Max Planck](https://en.wikipedia.org/wiki/Max_Planck) head bust using $n_{max} = 70$.

![](https://github.com/msshaqfa/spheroidal-harmonics/blob/main/visualizations/Max_example.gif)



