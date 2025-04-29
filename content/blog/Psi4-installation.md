+++
title = 'Psi4 Installation'
date = 2025-04-29T09:43:57-04:00
draft = false
+++


Started the [MOLSSI Quantum Chemistry class](https://www.udemy.com/course/introduction-to-quantum-chemistry-simulation/?referralCode=6859CB5EEE43242ACD2B). I’ve heard Psi4 in passing but I’ve not used it yet. 


Chose to install it with conda - avoiding any dependency issues. 
Tried others on the [psicode manual](https://psicode.org/psi4manual/master/external.html), which I had to eventually unistall and restart with the following: 

```bash
conda create -n psi4env -c conda-forge python=3.8 psi4
conda activate psi4env
```

this passed or skipped all of `psi4 --test` , but only after `conda install pytest`.

PS: I am on a 2020 Mac-M1.