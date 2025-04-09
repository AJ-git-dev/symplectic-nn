# Learning Area-Preserving 2D Embeddings for Symplectic Simulations

### Symplectic Embedding Neural Network Summary

This project serves as a toy model for exploring area-preserving diffeomorphisms in 2D, as a testbed for future symplectic embedding tools. It uses machine learning to train transformations from the unit disk in R^2 to other 2D regions while preserving total area. The core behavior mimics a key property of symplectic embeddings—area preservation—within a 2D framework (Schlenk, 2017, p. 10).

### Methodology
This project is inspired by the methods in *Optimization on Symplectic Embeddings* by Gajewski et al. Unlike their use of symplectic integrators applied to Hamiltonian flows, this implementation directly learns coordinate-wise mappings f: R^2 → R^2, constrained via the determinant of the Jacobian. While this does not enforce true symplecticity, it approximates one core geometric property of symplectic maps in 2D.

The main workflow is structured in five parts:

1. **Establishing the domain, D**  
   Sample uniformly from the 2D unit disk. D is written in ```utils.py```. 

2. **Defining the neural network map**  
   Train a small feedforward network to learn a transformation f(x, y) → (x', y'). The network is modeled in ```model.py```.  

3. **Loss function**  
   Penalize deviations from area preservation using the Jacobian determinant. Found in ```utils.py```.

4. **Training the network**  
   Optimized via PyTorch and Adam.   

5. **Visualization**  
   Display original and mapped shapes, along with area-preservation loss over time.

---

### Sample Output

Example Output Fig. 1: 

![Example Output Fig 1](https://raw.githubusercontent.com/AJ-git-dev/symplectic-nn/main/figures/example_output_fig1.png)

Example Output Fig. 2: 

![Example Output Fig 2](https://raw.githubusercontent.com/AJ-git-dev/symplectic-nn/main/figures/example_output_fig2.png)

To view the the details of the sample output of a run of the program, check the Jupyter notebook file `main.ipynb` and the `figures` folder. The notebook contains detailed steps of the computation and visualization, while the figures folder includes images generated during the execution of the program.

---

### Additional Information
- **Skills**: Python, NumPy, PyTorch  
- **Concepts**: 2-forms, area-preserving diffeomorphisms, Jacobian determinant

### References
- Gajewski, A., Goldin, E., Safin, J., Singh, N., & Zhang, J. (2019). *Optimization on symplectic embeddings*. https://kylersiegel.xyz/Optimization_on_Symplectic_Embeddings.pdf  
- Schlenk, F. (2017). *Symplectic embedding problems, old and new*. *Bulletin of the American Mathematical Society*, 55(2), 139–182. https://doi.org/10.1090/bull/1587
