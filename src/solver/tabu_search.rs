use crate::data::Instance;
use crate::solver::{
  generate_random_solution, get_orientation_from_schedule, n1, IntermediateSolution,
};
use log::{debug, info, trace};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_chacha;
use std::cmp;
use std::time::{Duration, Instant};

pub struct Config {
  pub timeout: Duration,
  pub seed: u64,
}

pub fn find_solution(inst: &Instance, config: &Config) -> IntermediateSolution {
  let mut rng = rand_chacha::ChaChaRng::seed_from_u64(config.seed);
  let mut current = generate_solution(inst, &mut rng);
  let mut best = current.clone();

  // Taillard, Parallel Taboo Search Techniques for the Job Shop Scheduling Problem
  let n = inst.n_jobs as f32;
  let m = inst.n_machines as f32;
  let tabu_duration =
    ((n + m / 2.0) * (-n / (5.0 * m)).exp() + (n * m) / 2.0 * (-5.0 * m / n).exp()) as i32;
  let mut op_last_swap = Array1::<i32>::from_elem(inst.n_ops(), i32::min_value());
  let mut op_push_back_count = Array1::<i32>::from_elem(inst.n_ops(), 0);
  let mut total_push_back_count = 0;
  // Maximum increase of cmax between two successive solutions
  let mut max_delta = 0;

  trace!("Starting with {}", current.cmax());
  let start = Instant::now();
  let mut iteration = 0;
  while Instant::now().duration_since(start) < config.timeout {
    let penalty_factor = 0.5 * max_delta as f32 * (n * m).sqrt();
    let maybe_move = n1::find_move(
      &current,
      &mut |maybe_best, candidate| {
        let (a, b) = candidate.swap_move;
        let tabu_until = op_last_swap[a] + tabu_duration;
        if iteration < tabu_until {
          // Aspiration criterion (globally better move)
          if candidate.cmax < best.cmax {
            trace!(
              "Including tabu move {:?} because it is better than global best {:?} < {:?}",
              candidate,
              candidate.cmax,
              best.cmax
            );
          } else {
            trace!(
              "Skipping move {:?} because it is tabu until {:?}",
              candidate,
              tabu_until
            );
            return false;
          }
        }

        if let Some(current_best) = maybe_best {
          let candidate_penalty =
            penalty_factor * op_push_back_count[b] as f32 / total_push_back_count as f32;
          let current_penalty = penalty_factor
            * op_push_back_count[current_best.swap_move.1] as f32
            / total_push_back_count as f32;
          let candidate_evaluation = candidate.cmax as f32 + candidate_penalty;
          let current_evaluation = current_best.cmax as f32 + current_penalty;
          candidate_evaluation < current_evaluation
        } else {
          true
        }
      },
      n1::SearchMethod::Exhaustive,
    );

    if let Some(next_move) = maybe_move {
      let swap_move = next_move.swap_move;
      let (a, b) = swap_move;
      let delta = next_move.cmax.saturating_sub(current.cmax);
      max_delta = cmp::max(max_delta, delta);

      current = current.apply_swap(a, b);
      op_last_swap[b] = iteration;
      op_push_back_count[b] += 1;
      total_push_back_count += 1;
      trace!(
        "Found move {:?} to {} ({})",
        swap_move,
        current.cmax,
        iteration
      );
      trace!("Current solution {:?} ({})", current, iteration);
      #[cfg(debug_assertions)]
      crate::solver::verify_solution(&inst, &current.to_solution()).expect("Verification failed");
    } else {
      debug!("Did not find move, resetting ({})", iteration);
      current = generate_solution(inst, &mut rng);
      op_last_swap.fill(i32::min_value());
      op_push_back_count.fill(0);
      total_push_back_count = 0;
      max_delta = 0;
    }

    if current.cmax() < best.cmax() {
      best = current.clone();
      debug!("Improved best to {} ({})", best.cmax(), iteration);
    }

    iteration += 1;
  }

  info!("Stopping due to timeout at {} ({})", best.cmax(), iteration);

  return best;
}

fn generate_solution<R: Rng>(inst: &Instance, rng: &mut R) -> IntermediateSolution {
  let orientation = get_orientation_from_schedule(inst, &generate_random_solution(inst, rng));

  return IntermediateSolution::new(inst.clone(), orientation);
}
