use crate::data::{Instance, Solution};
use crate::solver::{get_orientation_from_schedule, n1, IntermediateSolution};
use log::trace;

pub fn improve_solution(inst: &Instance, initial_solution: &Solution) -> IntermediateSolution {
  let mut current_solution = IntermediateSolution::new(
    inst.clone(),
    get_orientation_from_schedule(&inst, &initial_solution),
  );

  trace!("Starting with {}", current_solution.cmax());
  loop {
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
      trace!("Found improvement to {}", current_solution.cmax());
    } else {
      trace!(
        "Did not find improvement, stopping at {}",
        current_solution.cmax()
      );
      break;
    }
  }

  return current_solution;
}
