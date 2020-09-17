use crate::data::Instance;
use crate::solver::{
  generate_random_solution, get_orientation_from_schedule, n1, IntermediateSolution,
};
use log::{debug, info, trace};
use rand::seq::IteratorRandom;
use rand::{Rng, SeedableRng};
use rand_chacha;
use std::cmp;
use std::time::{Duration, Instant};

pub struct Config {
  pub timeout: Duration,
  pub seed: u64,
  pub start_acceptance_ratio: f64,
  pub delta: f64,
}

pub fn find_solution(inst: &Instance, config: &Config) -> IntermediateSolution {
  let mut rng = rand_chacha::ChaChaRng::seed_from_u64(config.seed);

  let mut best = generate_solution(inst, &mut rng);

  // Cruz-Chavez and Frausto-Solis, “Simulated Annealing with Restart to Job Shop Scheduling Problem Using Upper Bounds.”
  let start = Instant::now();
  let mut global_iteration = 0;
  while Instant::now().duration_since(start) < config.timeout {
    let improved = run_sa(inst, &mut rng, &mut global_iteration, &start, config);

    if improved.cmax() < best.cmax() {
      best = improved;
      debug!(
        "Improved global best to {} (iteration {})",
        best.cmax(),
        global_iteration
      );
    }

    global_iteration += 1;
  }

  info!(
    "Stopping due to timeout at {} (iteration {})",
    best.cmax(),
    global_iteration
  );

  return best;
}

fn run_sa<R: Rng>(
  inst: &Instance,
  rng: &mut R,
  global_iteration: &mut u64,
  start: &Instant,
  config: &Config,
) -> IntermediateSolution {
  let mut current = generate_solution(inst, rng);
  let mut current_neighborhood = n1::generate_moves(&current);
  let mut best = current.clone();

  let start_acceptance_ratio = config.start_acceptance_ratio;
  let delta = config.delta;
  // Size of largest possible neighborhood
  // From Van Laarhoven, Aarts, and Lenstra, “Job Shop Scheduling by Simulated Annealing.”
  let equilibrium_iterations = cmp::max(inst.n_ops().checked_sub(inst.n_machines).unwrap(), 1);

  // Aarts and Van Laarhoven, "Statistical Cooling."
  let initial_temperature = estimate_initial_temperature(inst, rng, start_acceptance_ratio);
  let mut temperature = initial_temperature;
  debug!(
    "Starting with cmax {}, temp {}, iterations {}",
    current.cmax(),
    temperature,
    equilibrium_iterations
  );
  while Instant::now().duration_since(*start) < config.timeout {
    let mut accepted_move_costs = vec![current.cmax()];
    for inner_iteration in 0..equilibrium_iterations {
      // Abort early if inner loop exceeds timeout
      if Instant::now().duration_since(*start) >= config.timeout {
        break;
      }

      if let Some(next_move) = current_neighborhood.iter().choose(rng) {
        let cost_next = next_move.cmax as f64;
        let cost_current = current.cmax() as f64;
        let cost_delta = cost_next - cost_current;
        let acceptance_threshold = if cost_delta <= 0.0 {
          1.0
        } else {
          f64::min(1.0, (-cost_delta / temperature).exp())
        };
        let should_accept_move = rng.gen_range(0.0, 1.0) < acceptance_threshold;
        if should_accept_move {
          let swap_move = next_move.swap_move;
          let (a, b) = swap_move;
          current = current.apply_swap(a, b);
          current_neighborhood = n1::generate_moves(&current);
          accepted_move_costs.push(current.cmax());
          trace!(
            "Accepted move {:?} to {} (iteration {}-{}, temp {})",
            swap_move,
            current.cmax(),
            global_iteration,
            inner_iteration,
            temperature
          );
          trace!(
            "Current solution {:?} (iteration {}={})",
            current,
            global_iteration,
            inner_iteration
          );
          #[cfg(debug_assertions)]
          crate::solver::verify_solution(&inst, &current.to_solution())
            .expect("Verification failed");
        } else {
          trace!(
            "Rejected move {:?} (iteration {}-{}, temp {})",
            next_move.swap_move,
            global_iteration,
            inner_iteration,
            temperature
          );
        }
      } else {
        // Should only happen when there are no candidates in the neighborhood e.g. for single machine problems
        debug!(
          "Did not find move, aborting (iteration {})",
          global_iteration
        );
        break;
      }
    }

    if current.cmax() < best.cmax() {
      best = current.clone();
      debug!(
        "Improved local best to {} (iteration {}, temp {})",
        best.cmax(),
        global_iteration,
        temperature
      );
    }

    // From Van Laarhoven, Aarts, and Lenstra, “Job Shop Scheduling by Simulated Annealing.”
    let accepted_move_costs_std_dev = std_dev(&accepted_move_costs)
      .expect("Unable to calculate std deviation for accepted_move_costs");

    if accepted_move_costs_std_dev.abs() > 0.0 {
      let sigma = accepted_move_costs_std_dev.abs();
      temperature = temperature / (1.0 + (temperature * (1.0 + delta).ln() / (3.0 * sigma)));
    } else {
      debug!(
        "Stopping because no more variation with temp {} at {} ({})",
        temperature,
        best.cmax(),
        global_iteration,
      );
      return best;
    }

    *global_iteration += 1;
  }

  debug!(
    "Stopping due to timeout at {} (iteration {})",
    best.cmax(),
    global_iteration
  );

  return best;
}

fn generate_solution<R: Rng>(inst: &Instance, rng: &mut R) -> IntermediateSolution {
  let orientation = get_orientation_from_schedule(inst, &generate_random_solution(inst, rng));

  return IntermediateSolution::new(inst.clone(), orientation);
}

fn mean(vec: &Vec<u32>) -> Option<f64> {
  let sum: f64 = vec.iter().map(|&x| x as f64).sum();
  let count = vec.len();

  return match count {
    0 => None,
    _ => Some(sum / count as f64),
  };
}

fn std_dev(vec: &Vec<u32>) -> Option<f64> {
  let mean = mean(vec)?;
  let count = vec.len();

  let sum_squared_delta: f64 = vec.iter().map(|x| (*x as f64 - mean).powi(2)).sum();
  return match count {
    0 => None,
    _ => Some(sum_squared_delta / count as f64),
  };
}

fn estimate_initial_temperature<R: Rng>(
  inst: &Instance,
  rng: &mut R,
  start_acceptance_ratio: f64,
) -> f64 {
  // Aarts, Korst, and van Laarhoven, “A Quantitative Analysis of the Simulated Annealing Algorithm.”
  let trials = 30;
  let mut deltas = Vec::new();
  for _ in 0..trials {
    let solution = generate_solution(inst, rng);
    let moves = n1::generate_moves(&solution);
    if let Some(chosen_move) = moves.iter().choose(rng) {
      let delta = chosen_move.cmax - solution.cmax;
      deltas.push(delta as f64);
    }
  }

  let (improving_moves, worsening_moves): (Vec<f64>, Vec<f64>) =
    deltas.into_iter().partition(|&d| d <= 0.0);
  let improving_move_count = improving_moves.len();
  let worsening_move_count = worsening_moves.len();

  let avg_positive_delta = worsening_moves.into_iter().sum::<f64>() / worsening_move_count as f64;
  let x0 = start_acceptance_ratio;
  let m1 = improving_move_count as f64;
  let m2 = worsening_move_count as f64;

  let c0 = avg_positive_delta / ((m2 / (m2 * x0 - (1.0 - x0) * m1)).ln());
  return c0;
}
