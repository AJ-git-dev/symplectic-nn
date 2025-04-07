# Learning Area-Preserving 2D Embeddings for Symplectic Simulations

### Summary
This project serves as a toy model for exploring area-preserving diffeomorphisms in 2D, as a testbed for future symplectic embedding tools. It uses machine learning to train transformations from the unit disk in \( \mathbb{R}^2 \) to other 2D regions while preserving total area. The core behavior mimics a key property of symplectic embeddings—area preservation—within a 2D framework (Schlenk, 2017, p. 10). Specifically, the neural network is trained to approximate maps \( f: \mathbb{R}^2 \to \mathbb{R}^2 \) under the constraint \( |\det(J_f(x))| \approx 1 \) (Gajewski et al., 2019, p. 21).

### Methodology
This project is inspired by the methods in *Optimization on Symplectic Embeddings* by Gajewski et al. Unlike their use of symplectic integrators applied to Hamiltonian flows, this implementation takes a simplified approach: it directly learns coordinate-wise mappings \( f: \mathbb{R}^2 \to \mathbb{R}^2 \), constrained via the determinant of the Jacobian. While this does not enforce true symplecticity, it approximates one core geometric property of symplectic maps in 2D.

The main workflow is structured in five parts:

1. **Establishing the domain \( D \)**  
   Sample uniformly from the 2D unit disk.

2. **Defining the neural network map**  
   Train a small feedforward network to learn a transformation \( f(x, y) \to (x', y') \).

3. **Loss function**  
   Penalize deviations from area preservation using the Jacobian determinant.

4. **Training the network**  
   Optimize via PyTorch and Adam.

5. **Visualization**  
   Display original and mapped shapes, along with area-preservation loss over time.

---

### Additional Information
- **Skills**: Python, NumPy, PyTorch  
- **Concepts**: 2-forms, area-preserving diffeomorphisms, Jacobian determinant

### References
- Gajewski, A., Goldin, E., Safin, J., Singh, N., & Zhang, J. (2019). *Optimization on symplectic embeddings*. https://kylersiegel.xyz/Optimization_on_Symplectic_Embeddings.pdf  
- Schlenk, F. (2017). *Symplectic embedding problems, old and new*. *Bulletin of the American Mathematical Society*, 55(2), 139–182. https://doi.org/10.1090/bull/1587
