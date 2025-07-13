+++
title = 'vdw-surfgen'
date = 2025-07-13T14:10:36-04:00
draft = false
tags = ["python", "chemistry", "molecular modelling", "vdw", "tools"]
+++


Wrote the titular package because we care about supramolecular chemistry.

I will come back someday to clarify the Fibonnacci sphere algorithm, but for now, the gist is that the code uses the generation of an evenly spaced point sphere around a group of points in space to generate the Van der-Waals surface for a molecule. 

It counts the atoms as the origin points and the atomic VdW surfaces as the spheres. Then it excludes crossing points between spheres to create the accurate molecular VdW surface. 

The scale of the points and the distance of the spheres from the origin points (which is an atom's 3D coordinate in this case) is crucial, hence the options. 

You can find the code and further information [here](https://github.com/sajagbe/vdw-surfgen) or install with `pip install vdw-surfgen`.


**Reference:**

1. [Stack Overflow](https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere#:~:text=The%20Fibonacci%20sphere%20algorithm%20is,a%20simple%20implementation%20in%20python.)

