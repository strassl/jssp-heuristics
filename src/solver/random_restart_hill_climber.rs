use crate::data::{Instance, Solution};
use crate::solver::{
  generate_random_solution, get_orientation_from_schedule, n1, IntermediateSolution,
};
use log::{debug, info, trace};
use rand::SeedableRng;
use rand_chacha;
use std::time::{Duration, Instant};

pub struct Config {
  pub timeout: Duration,
  pub seed: u64,
}

pub fn find_solution(inst: &Instance, config: &Config) -> IntermediateSolution {
  let mut rng = rand_chacha::ChaChaRng::seed_from_u64(config.seed);
  let mut current_solution = IntermediateSolution::new(
    inst.clone(),
    get_orientation_from_schedule(&inst, &generate_solution(&inst, &mut rng)),
  );
  let mut best_solution = current_solution.clone();

  trace!("Starting with {}", current_solution.cmax());
  let mut iteration = 0;
  let start = Instant::now();
  while Instant::now().duration_since(start) < config.timeout {
    let maybe_move = n1::find_move(
      &current_solution,
      &mut |maybe_best, candidate| {
        if let Some(best) = maybe_best {
          candidate.cmax < best.cmax
        } else {
          true
        }
      },
      n1::SearchMethod::Exhaustive,
    );
    let maybe_improvement = maybe_move.filter(|m| m.cmax < current_solution.cmax());

    if let Some(next_move) = maybe_improvement {
      let swap_move = next_move.swap_move;
      current_solution = current_solution.apply_swap(swap_move.0, swap_move.1);
      trace!(
        "Found improvement to {} ({})",
        current_solution.cmax(),
        iteration
      );
    } else {
      trace!(
        "Did not find improvement over {}, resetting ({})",
        current_solution.cmax(),
        iteration
      );
      current_solution = IntermediateSolution::new(
        inst.clone(),
        get_orientation_from_schedule(&inst, &generate_solution(&inst, &mut rng)),
      );
    }

    if current_solution.cmax() < best_solution.cmax() {
      best_solution = current_solution.clone();
      debug!(
        "Found global improvement to {} ({})",
        best_solution.cmax(),
        iteration
      );
    }

    iteration += 1;
  }

  info!(
    "Stopping due to timeout at {} ({})",
    best_solution.cmax(),
    iteration
  );

  return best_solution;
}

fn generate_solution<R: rand::Rng>(inst: &Instance, rng: &mut R) -> Solution {
  generate_random_solution(inst, rng)
}
