# (Meta-)Heuristics for the JSSP

## Build

Build using `cargo build --release`. The implementation was tested with rustc 1.44.1 (c7087fe00 2020-06-17) and LLVM 9.0.
The compiled binary is `target/release/heuristics`.

## Run

Run using `heuristic --instance <instance> --seed <seed> --solver <solver> --timeout <timeout>` where:

- `instance` is the path to the instance file in [standard specification](http://jobshop.jjvh.nl/explanation.php).
- `seed` is an arbitrary integer use to seed all randomized operations
- `timeout` is the maximum time in seconds an algorithm is permitted to run. All algorithms will usually slightly exceed this timeout since it is only checked after every iteration. Simple heuristics (i.e. `hill-climber`, `priority-*` and `sequential`) do not check the timeout at all.
- `solver` is the name of the solver to use. All metaheuristics use the neighborhood from [1]. Possible values are:

  - `hill-climber`: A best-improvement hill-climbing algorithm with an initial solution from `priority-sps`.
  - `random-restart-hill-climber`: A random-restart hill-climbing algorithm with a randomized initial solution.
  - `tabu-search`: A tabu-search algorithm based on [2].
  - `simulated-annealing`: A simulated annealing algorithm based on [1].
    Requires two additional parameters:
    - `sa-start-acceptance-ratio`: The initial acceptance ratio, used to derive the initial temperature.
    - `sa-delta`: Parameter controlling the cooling schedule.
  - `priority-sps`: A dispatching rule-based heuristic using the shortest processing sequence rule.
  - `priority-lps`: A dispatching rule-based heuristic using the longest processing sequence rule.
  - `priority-spt`: A dispatching rule-based heuristic using the shortest processing time rule.
  - `priority-lpt`: A dispatching rule-based heuristic using the longest processing time rule.
  - `priority-lwrm`: A dispatching rule-based heuristic using the least work remaining rule.
  - `priority-mwrm`: A dispatching rule-based heuristic using the most work remaining rule.
  - `sequential`: A sequential ordering of all operations.

The result is printed to stdout.
The first line contains the makespan of the solution, followed by a line for each job containing the start times of each operation.
All algorithms include logging output which can be turned on by setting the `RUST_LOG` environment variable e.g. `RUST_LOG="debug" heuristics [...]`.

## About

This work is part of Strassl, Simon. “Instance Space Analysis for the Job Shop Scheduling Problem.” Master’s Thesis, TU Vienna, 2020.

## References

[1] van Laarhoven, Peter J. M., Emile H. L. Aarts, and Jan Karel Lenstra. “Job Shop Scheduling by Simulated Annealing.” Operations Research 40, no. 1 (1992): 113–125. https://doi.org/10.1287/opre.40.1.113.

[2] Taillard, Eric D. “Parallel Taboo Search Techniques for the Job Shop Scheduling Problem.” ORSA Journal on Computing 6, no. 2 (1994): 108–117. https://doi.org/10.1287/ijoc.6.2.108.
