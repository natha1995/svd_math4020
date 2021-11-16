# svd_math4020
Singular Value Decomposition Final Project
Singular Value Decomposition in this project you will extract principal components from windows in the
time series and then interpret them.
Procedure:
1. Construct an 𝑁 × 𝐿 matrix [
− 𝑤1 −
⋮
− 𝑤𝑁 −
] and center the data to from a matrix 𝑋. 
The 𝐿 × 𝐿 covariance matrix 𝐶 =
𝑋⊤𝑋 / 𝑁−1
can be diagonalized 𝐶 = 𝑉𝑆𝑉⊤.
2. Factor 𝑋 = 𝑈Σ𝑉⊤, the columns of 𝑉 are the orthogonal principal axes and the columns of 𝑈Σ are
the principal components of 𝑋. What distribution do the principal components from? What does
this say about the data?
3. Plot the first few columns of 𝑉. Then for a given window 𝑤𝑖, find its principal components and plot
its approximation from the rank 𝑟 matrix 𝑋 = 𝑈Σ𝑟𝑉⊤ for 1 ≤ 𝑟 ≤ 5. In what sense are these
windows vectors in an 𝑟 dimensional space?
4. Note 𝐶 = 𝑉Σ𝑈⊤𝑈Σ𝑉⊤/(𝑛 − 1) = 𝑉Σ^2/𝑛−1 𝑉⊤, use this fact to make a logarithmic plot of the
percentage of total variance explained by each principal component. How many principal
components are relevant to analysis? After what point are they describing noise? Try to think of a
method to answer this question.
5. Group the windows by using some method of clustering on their principal components. Plots these
clusters in 2 or 3D.
6. For a section of the original time series, plot a sequence of windows colored by their principal
component cluster.
Note: For part 2 please do not use a high-level function to obtain the SVD, try to restrict yourself to
matrix operations and eigenvalue computations.
