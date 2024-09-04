# 2024 Summer School in Computational Biology @ UC

## Group Project: *Machine Learning approaches for finding structure in neural data*
[Project presentation](https://slides.com/renatocfduarte/cnc-template-be532f?token=L8bFFxRC)

In this project, you will be analyzing a synthetic dataset which represents the activity of cochlear nuclei when stimulated by
specific auditory input. The dataset was designed as a machine learning benchmark for spiking neural networks



### **Preliminaries:** Setting up project requirements and libraries

#### 1) Cloning the repository
[This repository](https://github.com/rcfduarte/computational-biology-2024) contains all you will need to work on the project. So, start by cloning it:
```shell
git clone git@github.com:rcfduarte/computational-biology-2024.git
```

If you have never worked with `git` or are unfamiliar with it, have a quick look at this [short introduction](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git).

**Note:** If you run into any trouble or are unable to clone the repository, please contact the course tutor.


#### 2) Setting up a virtual environment
As the project requires some specific libraries, it is recommended that you setup a [conda](https://docs.anaconda.com/) virtual environment. If you are
unfamiliar with the notion of virtual environments, don't get scared, all you will need to set it up is provided in the repository

```shell
conda env create -f conda_venv.yml
```


#### 3) Load dataset
To ensure everything is setup properly run `test_loader.py`. If it works properly, it should output something similar to the figure below:

![](./plots/sample.png)

---

### Tutorial & Example

Once you have everything set up and running, go through the tutorial notebook [tutorial_example.ipynb](tutorial_example.ipynb).

### Project

The tutorial ends with the application of Principal Component Analysis to the dataset. A natural extension is the application
of different dimensionality reduction and manifold learning algorithms.
In the project presentation and below, I leave some suggestions for algorithms you may want to try.

Ideally, you should compare their performance..