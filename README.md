# Meridian

**Meridian: Metric-Semantic Primitive Matching for Cross-View Geo-Localization Beyond Urban Environments** is a research project on cross-view global localization, matching high-level metric-semantic point and line primitives across aerial imagery and ground robot RGB-D data to localize a robot without any area-specific training.

This repository contains the source for the Meridian project website.

## Authors

- [Mason Peterson](https://github.com/mbpeterson70)<sup>*</sup>
- [Qingyuan Li](https://github.com/liqyn)<sup>*</sup>
- [Yixuan Jia](https://acl.mit.edu/people/yixuany)
- [Fernando Cladera](https://fcladera.com/)
- [Carlos Nieto-Granda](https://www.linkedin.com/in/carlos-nieto-granda-62446641/)
- [Camillo J. Taylor](https://www.cis.upenn.edu/~cjtaylor/home.html)
- [Jonathan P. How](https://www.mit.edu/~jhow/)

<sup>*</sup>Equal contribution

Published in IEEE Robotics and Automation Letters (RA-L), 2026.

## Links

- **Code**: [GitHub Repository](https://github.com/mit-acl/meridian)
- **Paper**: [arXiv](https://arxiv.org/abs/2606.06312)

## Abstract

Successful robot automation requires accurate global localization to support repeatability, task planning, goal specification, and safe operation. However, reliable localization in GNSS-denied environments remains an open problem. Overhead aerial imagery offers a promising solution, but existing approaches primarily target structured urban environments and have been rarely demonstrated in unstructured natural terrain. Limitations of the state-of-the-art include a reliance on models trained for specific environments, as well as difficulty handling repetitive geometries and featureless landscapes commonly found in natural outdoor areas. To overcome these challenges, we present Meridian, a method for matching high-level metric-semantic primitives across aerial images and ground robot RGB-D camera data that achieves accurate global localization and generalizes well across diverse environments, all without any training or algorithmic fine-tuning on area-specific data. We formulate novel consistency metrics to estimate a distribution over robot submap poses and to reject outlier hypotheses in a robust pose graph optimization step for accurate robot trajectory estimation. We demonstrate that our algorithm can localize a ground robot across a wide variety of environments, including an autonomous driving dataset, a park and campus area, and a wilderness camp, with an average optimized trajectory error of 2.5 m over 20 km of ground traversal.


# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
