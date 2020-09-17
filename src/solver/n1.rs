extern crate itertools;

use crate::data::OpId;
use crate::solver::{op_ordering, IntermediateSolution};
use log;
use std::collections::{BTreeSet, VecDeque};

pub enum SearchMethod {
  Exhaustive,
  First,
}

pub type SwapMove = (OpId, OpId);
#[derive(Debug, Clone)]
pub struct EvaluatedMove {
  pub swap_move: SwapMove,
  pub cmax: u32,
}

pub fn find_move(
  solution: &IntermediateSolution,
  should_accept: &mut dyn FnMut(&Option<EvaluatedMove>, &EvaluatedMove) -> bool,
  search_method: SearchMethod,
) -> Option<EvaluatedMove> {
  let moves = generate_moves(&solution);

  if log::log_enabled!(log::Level::Warn) {
    if moves.is_empty() {
      log::warn!("Generated neighborhood is empty");
    }
  }

  let mut best = None;
  for candidate_move in moves {
    log::trace!("Trying move {:?}", candidate_move);
    if should_accept(&best, &candidate_move) {
      log::trace!("Accepted move {:?}", candidate_move);
      best = Some(candidate_move);

      match search_method {
        SearchMethod::First => break,
        SearchMethod::Exhaustive => {}
      }
    }
  }

  log::trace!("best={:?}", best);

  return best;
}

pub fn generate_moves(solution: &IntermediateSolution) -> Vec<EvaluatedMove> {
  // Generate neighborhood by swapping critical orientations (on the longest path)
  // see Taillard, Parallel Taboo Search Techniques for the Job Shop Scheduling Problem and Van Laarhoven, Job shop scheduling by simulated annealing

  // Goal: Permute two successive and critical operations that use the same machine

  let mut critical_arcs = BTreeSet::new();
  let mut open = VecDeque::new();
  for op in 0..solution.instance.n_ops() {
    if solution.is_critical(op)
      && solution.succ_job[op] == None
      && solution.succ_machine[op] == None
    {
      open.push_back(op);
    }
  }
  log::trace!("critical_terminals={:?}", open);

  while let Some(current) = open.pop_front() {
    // We don't want to create branches since we are only interested in the longest path
    // If we allow branching we might create loops e.g.
    // 01 -> 02 -> 11 -> 12
    //  \ >------------> /
    // In the earlier graph the edge (01, 12) should not be included

    let critical_pre_job = solution.pre_job[current].filter(|&op| solution.is_critical(op));
    let critical_pre_machine = solution.pre_machine[current].filter(|&op| solution.is_critical(op));

    log::trace!(
      "Tracing from {:?} to {:?}({:?})[j] | {:?}({:?})[m]",
      current,
      critical_pre_job,
      solution.pre_job[current],
      critical_pre_machine,
      solution.pre_machine[current],
    );

    let mut nexts = Vec::new();
    match (critical_pre_job, critical_pre_machine) {
      (Some(o1), Some(o2)) => {
        match op_ordering(
          o1,
          o2,
          &solution.release_times,
          &solution.instance.durations,
        ) {
          std::cmp::Ordering::Less => nexts.push(o2),
          std::cmp::Ordering::Greater => nexts.push(o1),
          std::cmp::Ordering::Equal => {
            nexts.push(o1);
            nexts.push(o2);
          }
        }
      }
      (Some(o1), None) => nexts.push(o1),
      (None, Some(o2)) => nexts.push(o2),
      (None, None) => {}
    };

    // Only choose the later node
    for next in nexts {
      critical_arcs.insert((next, current));
      open.push_back(next);
    }
  }

  let mut moves = Vec::new();
  for &(a, b) in &critical_arcs {
    let swap = (a, b);
    // Swap with successor on same machine
    if solution.instance.machines[a] == solution.instance.machines[b]
      && solution.oriented_conflict_edges.contains(&swap)
    {
      let candidate_cmax = solution.cmax_after_swap(a, b);

      let candidate_move = EvaluatedMove {
        swap_move: swap,
        cmax: candidate_cmax,
      };
      moves.push(candidate_move);
    }
  }

  log::trace!("critical_arcs={:?}", critical_arcs);
  log::trace!("moves={:?}", moves);

  return moves;
}
