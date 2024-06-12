##  Exploring the Potential of Polynomial Basis Functions in Kolmogorov-Arnold Networks: A Comparative Study of Different Groups of Polynomials https://arxiv.org/abs/2406.02583

## Overview 


This packaege investigates the performance of 18 different polynomial basis functions, grouped into several categories based on their mathematical properties and areas of application. The study evaluates the effectiveness of these polynomial-based KANs on the MNIST dataset for handwritten digit classification.

## Key Features

- Explores the potential of 18 polynomial basis functions in KANs, including orthogonal polynomials, hypergeometric polynomials, q-polynomials, Fibonacci-related polynomials, combinatorial polynomials, and number-theoretic polynomials.
- Provides a structured overview of the characteristics and potential applications of each polynomial group.
- Compares the performance of polynomial-based KANs using metrics such as overall accuracy, Kappa, and F1 score.
- Analyzes the relationships between model complexity, number of parameters, training time, and overall performance.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the "Name_of_Function".ipynb` script to train and evaluate the polynomial-based KAN models on the MNIST dataset.

## Results

The study reveals that the Gottlieb-KAN model achieves the highest overall accuracy, Kappa, and F1 score among the evaluated polynomial-based KANs. However, further investigation and tuning of these polynomials on more complex datasets are necessary to fully understand their capabilities and potential in KANs.

## Future Work

- Applying the polynomial-based KANs to a wider range of datasets with varying complexity levels.
- Investigating the impact of different model architectures, optimization techniques, and hyperparameter settings on performance.
- Developing advanced analytical techniques to quantify the relative importance of various factors in KAN model performance.

## References

[1] A. N. Kolmogorov, "On the representation of continuous functions of several variables by superposition of continuous functions of one variable and addition," Doklady Akademii Nauk SSSR, vol. 114, pp. 953-956, 1957.

[2] Z. Liu et al., "KAN: Kolmogorov-Arnold Networks," arXiv preprint arXiv:2404.19756, 2024.

[3] S. SS, "Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation," arXiv preprint arXiv:2405.07200, 2024.

## License

This project is licensed under the MIT License.

## Acknowledgements

We would like to express our gratitude to the authors of the related works [1-3] for their valuable contributions and insights that inspired this comparative study of polynomial-based KANs.
## Cite 
@misc{seydi2024exploring,
      title={Exploring the Potential of Polynomial Basis Functions in Kolmogorov-Arnold Networks: A Comparative Study of Different Groups of Polynomials}, 
      author={Seyd Teymoor Seydi},
      year={2024},
      eprint={2406.02583},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
