Usage:

1. Generate sample observations from a sample trajectory of a sample CTSMC using 'generate_data.py'
    - the data file is found in 'evidence' with a unique ID

2. use 'run_forward_backward.py' on the generated file with any of the following settings

|  Action   |   Method   |      Equation       |    Grid    |                               Resulting in                               |
|-----------|------------|---------------------|------------|--------------------------------------------------------------------------|
| forward   | discrete   | don't care          | don't care | Standard HSMM forward algorithm (reference)                              |
| forward   | continuous | integrodifferential | don't care | Forward algorithm (paper)                                                |
| forward   | continuous | integral            | uniform    | Forward currents -> forward marginals (uniform grid, paper)              |
| forward   | continuous | integral            | adaptive   | Forward currents -> forward marginals (adaptive HSMM, paper)             |
| backward  | discrete   | don't care          | don't care | Standard HSMM backward algorithm                                         |
| backward  | continuous | integrodifferential | don't care | Backward algorithm (paper)                                               |
| backward  | continuous | integral            | uniform    | Backward currents -> backward marginal likelihood (uniform grid, paper)  |
| backward  | continuous | integral            | adaptive   | Backward currents -> backward marginal likelihood (adaptive HSMM, paper) |
| posterior | discrete   | don't care          | don't care | Standard HSMM posterior marginal inference                               |
| posterior | continuous | don't care          | don't care | Posterior marginal inference from forward/backward currents (paper)      |

    - the results file is found under 'results_raw/forward_backward/<ID>' with proper naming

3. use 'process_forward_backward_results.py' on the results file to generate human readable results, plots, machine stats etc.
    - the generated files are found under 'results/forward_backward/<ID>' with proper naming
